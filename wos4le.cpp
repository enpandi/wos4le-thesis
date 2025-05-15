#include <polyscope/polyscope.h>
#include <polyscope/surface_mesh.h>
#include <polyscope/point_cloud.h>

#include <igl/readOFF.h>
#include <igl/AABB.h>
#include <igl/signed_distance.h>

#include <Eigen/Dense>

#include <iostream>
#include <thread>
#include <mutex>

class Mesh {
	// acceleration structures, similar to bounding volume hierarchy
	igl::AABB<Eigen::MatrixX3d, 3> tree;
	igl::WindingNumberAABB<Eigen::RowVector3d, Eigen::MatrixX3d, Eigen::MatrixX3i> hier;

public:
	// vertices, Nx3 matrix of doubles
	Eigen::MatrixX3d V;
	// triangle faces, Nx3 matrix of vertex index ints
	Eigen::MatrixX3i F;

	Mesh(const std::string& path_to_off) {
		igl::readOFF(path_to_off, V, F);
		tree.init(V, F);
		hier.set_mesh(V, F);
	}
	void ClosestPointQuery(const Eigen::Vector3d& p, double& distance, Eigen::Vector3d& closest_point) const {
		int closest_face;
		Eigen::RowVector3d closest_point_transposed;
		distance = std::sqrt(tree.squared_distance(V, F, p.transpose(), closest_face, closest_point_transposed));
		closest_point = closest_point_transposed.transpose();
	}
	bool ContainsPoint(const Eigen::Vector3d& p) const {
		Eigen::RowVector3d p_transpose = p.transpose();
		return igl::signed_distance_winding_number(tree, V, F, hier, p_transpose) <= 0.0;
	}
};

/*
 * Welford's online algorithm for means/variances
 * https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford%27s_online_algorithm
 * computes means/variances for three independent components.
 * also tracks the distribution of the number of steps taken in a walk.
 * NOT THREAD-SAFE
 */
struct Stats {
	uint64_t count;
	Eigen::Vector3d mean, _m2;
	std::vector<int> steps_taken_dist;
	Stats() { count = 0; mean.setZero(); _m2.setZero(); steps_taken_dist.clear(); }
	void Update(const Eigen::Vector3d& newval, int steps_taken) {
		++count;
		Eigen::Vector3d delta = newval - mean;
		mean += delta / count;
		Eigen::Vector3d delta2 = newval - mean;
		_m2 += delta.cwiseProduct(delta2);

		if (steps_taken >= steps_taken_dist.size()) steps_taken_dist.resize(steps_taken + 1);
		++steps_taken_dist[steps_taken];
	}
	Eigen::Vector3d Variance() {
		if (count < 2) return Eigen::Vector3d::Constant(NAN);
		return _m2 / (count - 1);
	}
};

// random number generators
namespace Random {
	thread_local std::mt19937 gen{std::random_device{}()};
	thread_local std::uniform_real_distribution<double> uniform_dist;
	thread_local std::normal_distribution normal_dist{};
	double Uniform() { return uniform_dist(gen); }
	double Gaussian() { return normal_dist(gen); }
	Eigen::Vector3d UnitSphere() {
		Eigen::Vector3d h(Gaussian(), Gaussian(), Gaussian());
		h.normalize();
		return h;
	}
	Eigen::Vector3d UnitBall() {
		Eigen::Vector3d s;
		do s << Uniform()*2-1, Uniform()*2-1, Uniform()*2-1;
		while (s.squaredNorm() > 1.0);
		return s;
	}
}

struct WalkParameters {
	// the point whose displacement we are estimating
	Eigen::Vector3d p{0.1, 0.0, 0.0};
	// ignore points outside mesh
	bool mesh_contains_p{false};

	// stiffness parameters, stopping tolerance
	double lambda{0.0}, mu{1.0}, epsilon{1e-3};

	// stretched boundary condition coefficients
	Eigen::Vector3d s{2.0, 0.0, 0.0};

	double ComputeB() const { return (lambda + 2*mu)  / (lambda + mu); }

	Eigen::Vector3d BoundaryDisplacement(const Eigen::Vector3d& p) const { return s.cwiseProduct(p); };

	Eigen::Vector3d TrueDisplacement() const {
		/* only works for linear stretching conditions */
		return BoundaryDisplacement(p);
	}

	void UpdateMeshContainsP(const Mesh& mesh) { mesh_contains_p = mesh.ContainsPoint(p); }

	bool operator==(const WalkParameters& wp) const {
		return
			p == wp.p && mesh_contains_p == wp.mesh_contains_p
			&& lambda == wp.lambda && mu == wp.mu
			&& epsilon == wp.epsilon
			&& s == wp.s;
	}
	bool operator!=(const WalkParameters& wp) const { return !(*this == wp); }
};

int main(int argc, char *argv[]) {


	// input domain of interest
	if (argc < 2) return std::cerr << "need to provide an OFF mesh file\n", 1;
	const Mesh mesh(argv[1]);


	// simulation parameters/outputs
	std::mutex mutex;
	WalkParameters walk_params;
	walk_params.UpdateMeshContainsP(mesh);
	Stats stats;


	// Algorithm 1
	auto linearElasticityWalkOnSpheres = [&mesh](const WalkParameters& walk_params, int& num_steps) -> Eigen::Vector3d {
		if (!walk_params.mesh_contains_p) return Eigen::Vector3d::Constant(NAN);
		const double&
			epsilon = walk_params.epsilon,
			lambda = walk_params.lambda,
			mu = walk_params.mu,
			b = walk_params.ComputeB();
		Eigen::Vector3d p = walk_params.p;
		Eigen::Matrix3d multiplier; multiplier.setIdentity();

		for (num_steps = 0;; ++num_steps) {
			double distance;
			Eigen::Vector3d closest_point;
			mesh.ClosestPointQuery(p, distance, closest_point);
			if (distance < epsilon)
				return walk_params.BoundaryDisplacement(closest_point);
			if (Random::Uniform() < 0.5) {
				Eigen::Vector3d h = Random::UnitSphere();
				multiplier *= ((2*b-1)*mu*Eigen::Matrix3d::Identity() + (2*b*(lambda+mu)+mu)*h*h.transpose()) / (b*mu);
				p += distance * h;
			} else {
				Eigen::Vector3d s = Random::UnitBall();
				multiplier *= -2.0*(lambda+mu)/(3*b*mu*s.squaredNorm()) * (b*Eigen::Matrix3d::Identity() + (1-3*b)*s*s.transpose()/s.squaredNorm());
				p += distance * s;
			}
		}
	};


	// spawn workers to run algorithm 1
	bool threads_done = false;
	std::vector<std::thread> threads;
	for (int i = 0; i < std::thread::hardware_concurrency(); ++i)
		threads.emplace_back([&]() {
			while (!threads_done) {
				WalkParameters walk_params_copy;
				{
					std::lock_guard guard{mutex};
					walk_params_copy = walk_params;
				}
				if (!walk_params_copy.mesh_contains_p || walk_params_copy.epsilon <= 0.0) {
					std::this_thread::sleep_for(std::chrono::milliseconds(100));
					continue;
				}

				int num_steps{};
				Eigen::Vector3d single_sample_u_p_est = linearElasticityWalkOnSpheres(walk_params_copy, num_steps);
				{
					std::lock_guard guard{mutex};
					// check if params are outdated
					if (walk_params == walk_params_copy) {
						stats.Update(single_sample_u_p_est, num_steps);
					}
				}
			}
		});


	// gui
	polyscope::init();
	polyscope::options::automaticallyComputeSceneExtents = false;

	auto psSlicePlane = polyscope::addSceneSlicePlane();
	psSlicePlane->setDrawWidget(true);
	psSlicePlane->setPose(glm::vec3(-1,0,0), glm::vec3(0,0,-1));

	polyscope::registerSurfaceMesh("rest domain", mesh.V, mesh.F)->setTransparency(0.5);
	auto DrawDeformedOmega = [&mesh](const WalkParameters& walk_params) {
		decltype(mesh.V) deformedV = mesh.V;
		for (int row = 0; row < deformedV.rows(); ++row)
			deformedV.row(row).transpose() += walk_params.BoundaryDisplacement(deformedV.row(row).transpose());
		polyscope::registerSurfaceMesh(
			"deformed domain",
			deformedV,
			mesh.F)->setTransparency(0.5);
	};
	DrawDeformedOmega(walk_params);

	auto psPointCloud
		= polyscope::registerPointCloud("sample point p", walk_params.p.transpose());
	psPointCloud
		->setPointRadius(0.001)
		->setPointRenderMode(polyscope::PointRenderMode::Quad)
		->setIgnoreSlicePlane(psSlicePlane->name, true);
	auto DrawP = [&psPointCloud](const WalkParameters& walk_params) {
		psPointCloud->updatePointPositions(walk_params.p.transpose());
		psPointCloud->addVectorQuantity(
			"actual/analytic displacement vector u(p) "
			"(implemented only for linear stretch boundary conditions)",
			walk_params.TrueDisplacement().transpose()
		);
	};
	DrawP(walk_params);

	polyscope::state::userCallback = [&]() {
		WalkParameters walk_params_copy;
		{
			std::lock_guard guard{mutex};
			walk_params_copy = walk_params;
		}
		static bool logging_enabled{};
		bool reset_estimate{};
		if (ImGui::Button("reset walk params")) {
			walk_params_copy = decltype(walk_params_copy){};
			walk_params_copy.UpdateMeshContainsP(mesh);
		}
		ImGui::InputDouble("lambda", &walk_params_copy.lambda);
		ImGui::InputDouble("mu", &walk_params_copy.mu);
		ImGui::InputDouble("epsilon", &walk_params_copy.epsilon);
		if (
			ImGui::InputDouble("p.x", &walk_params_copy.p[0])
			| ImGui::InputDouble("p.y", &walk_params_copy.p[1])
			| ImGui::InputDouble("p.z", &walk_params_copy.p[2])
		) {
			walk_params_copy.UpdateMeshContainsP(mesh);
			DrawP(walk_params_copy);
		}
		if (
			ImGui::InputDouble("s1", &walk_params_copy.s[0])
			| ImGui::InputDouble("s2", &walk_params_copy.s[1])
			| ImGui::InputDouble("s3", &walk_params_copy.s[2])
		) {
			DrawDeformedOmega(walk_params_copy);
		}
		if (!logging_enabled && ImGui::Button("reset estimate and start logging")) {
			logging_enabled = true;
			reset_estimate = true;
		}
		Stats stats_copy;
		{
			std::lock_guard guard{mutex};
			if (walk_params != walk_params_copy || reset_estimate) stats = decltype(stats){};
			walk_params = walk_params_copy;
			stats_copy = stats;
		}
		psPointCloud->addVectorQuantity(
			"estimated displacement vector u(p)",
			stats_copy.mean.transpose()
		)->setEnabled(true);
		if (logging_enabled) {
			std::cout
				<< (stats_copy.mean - walk_params_copy.TrueDisplacement()).norm() << ", "
				<< stats_copy.Variance().norm() << ", "
				<< stats_copy.count << '\n';
		}
	};
	polyscope::show();


	// clean up
	threads_done = true;
	for (auto& thread : threads) thread.join();

	std::cout << "steps_taken_distribution=[";
	for (const auto& freq : stats.steps_taken_dist)
		std::cout << freq << ',';
	std::cout << ']' << std::endl;
}












#if 0
// Eigen3
#include <Eigen/Dense>
#include <Eigen/Sparse>

// polyscope
#include <polyscope/polyscope.h>
#include <polyscope/surface_mesh.h>
#include <polyscope/point_cloud.h>
#include <polyscope/floating_quantity_structure.h>
#include <polyscope/scalar_render_image_quantity.h>
#include <polyscope/curve_network.h>
#include <polyscope/implicit_helpers.h>
#include <polyscope/volume_mesh.h>


// libigl
#include <igl/readMESH.h>
#include <igl/readOFF.h>
#include <igl/unproject_on_plane.h>
#include <igl/signed_distance.h>
#include <igl/pseudonormal_test.h>
#include <igl/edges.h>
#include <igl/AABB.h>
#include <igl/boundary_loop.h>
#include <igl/barycentric_coordinates.h>
#include <igl/per_vertex_normals.h>
#include <igl/per_edge_normals.h>

// STL
#include <iostream>
#include <mutex>
#include <thread>
#include <random>
#include <cstdlib>
#include <map>
// #include <numbers> // C++20
#include <set>


namespace Random {
	thread_local std::random_device rd{};
	thread_local std::mt19937 gen{rd()};
	thread_local std::normal_distribution normal_dist{};
	thread_local std::uniform_real_distribution<double> uniform_dist;

	double uniform() { return uniform_dist(gen); }

	double gaussian() { return normal_dist(gen); }

	Eigen::Vector3d unitSphere() {
		// random unit vector (uniform over the sphere)
		Eigen::Vector3d res; res << gaussian(), gaussian(), gaussian();
		res.normalize();
		return res;
	}

	Eigen::Vector3d unitBall() {
		// random vector with norm <= 1
		Eigen::Vector3d res;
		do res << uniform()*2-1, uniform()*2-1, uniform()*2-1;
		while (res.squaredNorm() > 1);
		return res;
	}
}

namespace Mesh {
	Eigen::MatrixX3d V, FN, VN, EN;
	Eigen::MatrixX3i F;
	Eigen::MatrixX2i E;
	Eigen::VectorXi EMAP;
	igl::AABB<Eigen::MatrixX3d, 3> tree;
	igl::WindingNumberAABB<Eigen::RowVector3d, Eigen::MatrixX3d, Eigen::MatrixX3i> hier;
	// igl::FastWindingNumberBVH fwn_bvh;
	// polyscope::SurfaceMesh* psMesh{};

	void init() {
		// Set V and F before calling Mesh::init()
		static bool is_initialized = false; assert(!is_initialized); is_initialized = true;
		assert(V.rows() && F.rows());
		tree.init(V, F);
		hier.set_mesh(V, F);
		igl::per_face_normals(V, F, FN);
		igl::per_vertex_normals(V, F, igl::PER_VERTEX_NORMALS_WEIGHTING_TYPE_ANGLE, FN, VN);
		igl::per_edge_normals(V, F, igl::PER_EDGE_NORMALS_WEIGHTING_TYPE_UNIFORM, FN, EN, E, EMAP);
		// igl::fast_winding_number(V, F, 2/*Taylor series expansion order to use (e.g., 2)*/, fwn_bvh);
#if 0
		psMesh = polyscope::registerSurfaceMesh("mesh", V, F);
		std::vector<double> ub(V.rows());
		for (int i = 0; i < V.rows(); ++i) ub[i] = boundaryValue(V.row(i).transpose());
		u_b = psMesh->addVertexScalarQuantity("u_b", ub);
		u_b->setEnabled(true);
#endif
	}

	double sdf(Eigen::Vector3d p) {
		// + for outside, - for inside
		Eigen::RowVector3d q = p.transpose();
		return igl::signed_distance_winding_number(tree, V, F, hier, q);
		// return igl::signed_distance_fast_winding_number(q, V, F, tree, fwn_bvh);
		// return igl::signed_distance_pseudonormal(tree, V, F, FN, VN, EN, EMAP, q);
	}
}

struct LinearElasticity {
	double lambda, mu; // Lamé parameters
	double b; // for Thomson's/Kelvin's solution

	LinearElasticity(double lambda, double mu): lambda{lambda}, mu{mu} {
#if 1
		b = (lambda+2*mu) / (lambda+mu);
#else
		const double bulk_modulus_K = lambda + mu*2/3;
		const double poissons_ratio_nu = lambda / (bulk_modulus_K*3 - lambda);
		const double a = 1 - 2*poissons_ratio_nu;
		b = 2 * (1 - poissons_ratio_nu);
#endif
	}

	/*
	https://en.wikipedia.org/wiki/Linear_elasticity#Thomson's_solution_-_point_force_in_an_infinite_isotropic_medium
	Thomson's solution - point force in an infinite isotropic medium
	The most important solution of the Navier–Cauchy or elastostatic
	equation is for that of a force acting at a point in an infinite
	isotropic medium. This solution was found by William Thomson
	(later Lord Kelvin) in 1848 (Thomson 1848). This solution is the
	analog of Coulomb's law in electrostatics. A derivation is given
	in Landau & Lifshitz.[7]: §8 
	*/

	/*
	LinearElasticity::G returns Green's tensor centered at x, evaluated at y.
	x and y should be the same as Sawhney & Crane's convention.
	*/
	Eigen::Matrix3d G(const Eigen::Vector3d x, const Eigen::Vector3d y) const {
		const Eigen::Vector3d p = y - x;
		const double r2 = p.squaredNorm();
		const double r2_times_2b = r2 * 2 * b;
		const double one_minus_recip_2b = 1 - 1 / (2 * b);
		const double four_pi_mu_r = 4 * M_PI * mu * std::sqrt(r2);
		Eigen::Matrix3d G_p;
		for (int i = 0; i < 3; ++i) for (int k = 0; k < 3; ++k) {
			G_p(i, k) = p(i) * p(k) / r2_times_2b;
			if (i != k) continue;
			G_p(i, k) += one_minus_recip_2b;
		}
		G_p /= four_pi_mu_r;
		return G_p;
	}

	/*
	Let i be a column index in 1, 2, 3,
	G_i(x, y) be the i-th column of the Thomson solution Green's tensor,
	and n(x, y) = (y - x) / |y - x| be the outward normal vector.

	The boundary integral equation for the i-th component of u(x) consists of two terms:
	an integral over the solid ball, and an integral over the hollow spherical shell.

	Each integrand can be written in the form
	f(lambda, mu, G_i(x, y), n(x, y)) dot u(y).

	LinearElasticity::SolidIntegrand(x, y) and LinearElasticity::HollowIntegrand(x, y)
	return matrices whose i-th rows correspond to their respective
	f(lambda, mu, G_i(x, y), n(x, y)).

	With these definitions, u(x) is a sum of integrals of H u(y) dy
	for H returned by these functions.

	Note also that H is symmetric for both integrands.
	*/

	/*
	Sympy:
	simp_solidG = (Matrix([x,y,z])*Matrix([x,y,z]).T * (1-3*b) + eye(3)*b*r2) * (lamda+mu) / (4*pi*b*mu*r2*r2)
	simp_hollowG = (Matrix([x,y,z])*Matrix([x,y,z]).T * (2*b*(lamda+mu)+mu) + eye(3) * mu*(2*b-1)*r2) / (-8*pi*b*mu*r2*sqrt(r2))
	*/

	// the H and S matrices designed to avoid multiplying by R
	Eigen::Matrix3d HollowMultiplier(const Eigen::Vector3d h) const {
		Eigen::Matrix3d hm= ((2*b-1)*mu*Eigen::Matrix3d::Identity() + (2*b*(lambda+mu)+mu)*h*h.transpose()) / (b*mu);
		//std::cerr<<hm.determinant()<<"hm\n\n";
		return hm;
	}
	Eigen::Matrix3d SolidMultiplier(const Eigen::Vector3d s) const {
		Eigen::Matrix3d sm= -2*(lambda+mu)/(3*b*mu*s.squaredNorm()) * (b*Eigen::Matrix3d::Identity() + (1-3*b)*s*s.transpose()/s.squaredNorm());
		//std::cerr<<sm.determinant()<<"sm\n\n";
		return sm;
	}

	// the full H and S matrices
	Eigen::Matrix3d Hfull(const Eigen::Vector3d p) const {
		Eigen::Matrix3d H = (
			(2*b-1)*mu*Eigen::Matrix3d::Identity()
			+ (2*b*(lambda+mu)+mu)/p.squaredNorm()*p*p.transpose()
		) / (8*M_PI*b*mu*p.norm());
		return H;
	}
	Eigen::Matrix3d Sfull(const Eigen::Vector3d p) const {
		Eigen::Matrix3d S = (
			b*Eigen::Matrix3d::Identity()
			+ (1-3*b)/p.squaredNorm()*p*p.transpose()
		) * (lambda+mu) / (4*M_PI*b*mu*p.squaredNorm());
		return S;
	}
};

namespace Util {
	// Welford's online algorithm
	// https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford%27s_online_algorithm
	template<class T> struct WelfordMeanScalar {
		uint64_t count{};
		T mean{}, m2{};
		void update(T newval) {
			++count;
			T delta = newval - mean;
			mean += delta / count;
			T delta2 = newval - mean;
			m2 += delta * delta2;
		}
		T getMean() { return mean; }
		T getVar() { return m2 / (count - 1); }
		uint64_t getCount() { return count; }
	};
	template<class T> struct WelfordMeanEigen {
		uint64_t count{};
		T mean{T::Zero()}, m2{T::Zero()};
		void update(T newval) {
			++count;
			T delta = newval - mean;
			mean += delta / count;
			T delta2 = newval - mean;
			m2 += delta.cwiseProduct(delta2);
		}
		T getMean() { return mean; }
		T getVar() { return m2 / (count - 1); }
		uint64_t getCount() { return count; }
	};
}

// compare to https://libigl.github.io/tutorial/#a-quick-primer-on-linear-elastostatics-25
int main(int argc, char *argv[]) {
	if (argc < 2) {
		std::cerr << "need to provide a mesh file\n";
		return 1;
	}

	polyscope::init();
	polyscope::options::warnForInvalidValues = false;

	Eigen::MatrixX3d V;
	Eigen::MatrixX3i F;
	igl::readOFF(argv[1], V, F);
	const size_t N = V.rows();
	std::unordered_set<int> boundary_vertices(
		F.reshaped().begin(), F.reshaped().end());
	Mesh::V = V;
	Mesh::F = F;
	Mesh::init();
	const double min_coordinate = V.minCoeff(), max_coordinate = V.maxCoeff();
	Eigen::RowVector3d min=V.row(0),max=V.row(0);
	for(int i=0;i<V.rows();++i){
		Eigen::RowVector3d row=V.row(i);
		for(int j=0;j<3;++j){
			min(j)=std::min(min(j),row(j));
			max(j)=std::max(max(j),row(j));
		}
	}

	polyscope::registerSurfaceMesh("original mesh", V, F)->setTransparency(0.5);
	polyscope::SurfaceMesh* psMesh = polyscope::registerSurfaceMesh("mesh", V, F);

	polyscope::SlicePlane* psSlicePlane = polyscope::addSceneSlicePlane();
	psSlicePlane->setDrawWidget(true);

	polyscope::PointCloud* psPointCloud = polyscope::registerPointCloud("sample point p", Eigen::RowVector3d::Zero())
		->setPointRadius(0.01)
		->setPointRenderMode(polyscope::PointRenderMode::Quad);

	constexpr double epsilon = 1e-2;
 	const LinearElasticity le(0.0, 1.0);
	std::mutex u_p_mutex;
	const Eigen::Vector3d p(0.2, 0.0, 0.0);
	Util::WelfordMeanEigen<Eigen::Vector3d> u_p;
	// displacement vector defined on boundary
	const Eigen::DiagonalMatrix<double, 3> stretch(2.0, 0.0, 0.0);
	const Eigen::Matrix3Xd u_b = stretch * V.transpose();
	const Eigen::Vector3d u_p_expected = stretch * p;

	psMesh->updateVertexPositions(V + u_b.transpose());
	psPointCloud->updatePointPositions(p.transpose());

	#if 0
	auto reset_inputs = [&]() {
		input_lambda = 0.0;
		input_mu = 1.0;
		input_epsilon = 1e-3;
		std::fill(input_p, input_p + 3, 0.0);
		std::fill(input_bnd_stretch, input_bnd_stretch + 3, 0.0);
		input_p[0] = 0.1; input_bnd_stretch[0]=2.0;
	};
	reset_inputs();
	auto set_params = [&]() {
		le = LinearElasticity(input_lambda, input_mu);
		p = Eigen::Vector3d(input_p[0], input_p[1], input_p[2]);;
		u_p = decltype(u_p)();
		u_b = Eigen::DiagonalMatrix<double, 3>(input_bnd_stretch[0], input_bnd_stretch[1], input_bnd_stretch[2]) * V.transpose();
		psMesh->updateVertexPositions(V + u_b.transpose());
		psPointCloud->updatePointPositions(p.transpose());
	};
	set_params();
#endif

	auto one_walk = [&](const Eigen::Vector3d& p) -> Eigen::Vector3d {
	again:
		Eigen::Vector3d xk = p;
		Eigen::Matrix3d multiplier = Eigen::Matrix3d::Identity();
		int steps = 0;
		constexpr int STEP_LIMIT = 0;
		for (; steps < STEP_LIMIT; ++steps) {
			double r = -Mesh::sdf(xk);
			// double r = 1 - xk.norm(); // SPHERE ONLY!
			if (r < 0) {
				std::cerr << "r<0!\n";
				std::this_thread::sleep_for(std::chrono::milliseconds(100));
			}
			assert(r > 0);
			if (r < epsilon) break;
#if 0
			// no simplification
			if (Random::uniform() < 0.5) {
				// W_H hollow sample
				double ball_surface_area = 4 * M_PI * r * r;
				Eigen::Vector3d qH = xk + Random::unitSphere() * r;
				multiplier *= 2 * ball_surface_area / r * le.Hfull(qH - xk);
				xk = qH;
			} else {
				// W_S solid sample
				double ball_volume = 4 * M_PI * r * r * r / 3;
				Eigen::Vector3d qS = xk + Random::unitBall() * r;
				multiplier *= -2 * ball_volume / r * le.Sfull(qS - xk);
				xk = qS;
			}
#else
			// simplified
			if (Random::uniform() < 0.5) {
				// W_H hollow sample
				Eigen::Vector3d h = Random::unitSphere();
				multiplier *= le.HollowMultiplier(h);
				xk += r * h;
			} else {
				// W_S solid sample
				Eigen::Vector3d s = Random::unitBall();
				multiplier *= le.SolidMultiplier(s);
				xk += r * s;
			}
#endif
		}
		if (steps!=STEP_LIMIT)goto again;
		// Barycentrically interpolate the Dirichlet boundary values
#if 1
		return multiplier * stretch * xk;
#else
		int closest_face;
		Eigen::RowVector3d closest_point;
		Mesh::tree.squared_distance(V, F, xk.transpose(), closest_face, closest_point);
		Eigen::RowVector3i face_vertices = F.row(closest_face);
		Eigen::RowVector3d barycentric_coordinates;
		igl::barycentric_coordinates(
			closest_point,
			V.row(face_vertices(0)),
			V.row(face_vertices(1)),
			V.row(face_vertices(2)),
			barycentric_coordinates
		);
		Eigen::Vector3d u0_closest = 
			barycentric_coordinates(0) * u_b.col(face_vertices(0))
			+ barycentric_coordinates(1) * u_b.col(face_vertices(1))
			+ barycentric_coordinates(2) * u_b.col(face_vertices(2));
		Eigen::Vector3d u_y = multiplier * u0_closest;
		std::cerr << u_y.transpose() << '\n';
		return u_y;
#endif
	};
	bool threads_done = false;
	std::vector<std::thread> threads;
	for (int i = 0; i < std::thread::hardware_concurrency(); ++i) threads.emplace_back([&]() {
		while (!threads_done) {
			if (-Mesh::sdf(p) < 0) {
				std::cerr << "p=" << p.transpose() << " is not in the mesh\n";
				std::this_thread::sleep_for(std::chrono::milliseconds(100));
				continue;
			}
			Eigen::Vector3d u_p_one_walk = one_walk(p);
			{
				std::lock_guard<std::mutex> u_p_guard{u_p_mutex};
				u_p.update(u_p_one_walk);
			}
		}
	});

	polyscope::state::userCallback = [&]() {
		#if 0
		if (ImGui::Button("reset")) reset_inputs(), set_params();
		if (ImGui::Button("reset u(p)")) {
			std::lock_guard<std::mutex> u_p_guard{u_p_mutex};
			u_p = decltype(u_p)();
		}
		/*
		Although the shear modulus, μ, must be positive,
		the Lamé's first parameter, λ, can be negative, in principle;
		however, for most materials it is also positive.
		*/
		if (
			ImGui::InputDouble("Lamé's first parameter λ", &input_lambda)
			||
			ImGui::InputDouble("Lamé's second parameter μ", &input_mu)
			||
			ImGui::InputDouble("WoS stopping tolerance ε", &input_epsilon)
			||
			ImGui::SliderScalarN("sample position p", ImGuiDataType_Double, input_p, 3, &min_coordinate, &max_coordinate)
			||
			ImGui::SliderScalarN("boundary stretch factor", ImGuiDataType_Double, input_bnd_stretch, 3, &MIN_BND_STRETCH, &MAX_BND_STRETCH)
		)
			set_params();
		#endif
		Eigen::Vector3d u_p_mean, u_p_var; uint64_t u_p_cnt;
		{
			std::lock_guard<std::mutex> u_p_guard{u_p_mutex};
			u_p_mean = u_p.getMean();
			u_p_var = u_p.getVar();
			u_p_cnt = u_p.getCount();
		}
		Eigen::Vector3d u_p_err = u_p_mean - u_p_expected;
		if (0)
			std::cerr << "n=" << u_p_cnt
				<< " mean=" << u_p_mean.transpose()
				<< " |err|=" << u_p_err.norm()
				<< " |var|=" << u_p_var.norm() << '\n';
		else
			std::cout << u_p_cnt << ',' << u_p_err.norm() << ',' << u_p_var.norm() << '\n';
		polyscope::PointCloudVectorQuantity* psPointCloudVectorQuantity = psPointCloud->addVectorQuantity("u(p)", u_p_mean.transpose(), polyscope::VectorType::AMBIENT)
			->setVectorRadius(0.1, false);
			//->setVectorLengthScale(1.0, false);
		psPointCloudVectorQuantity->setEnabled(true);
		psPointCloud->setIgnoreSlicePlane(psSlicePlane->name, true);
		if (u_p_cnt > 10000000) { std::cout.flush(); polyscope::unshow(); }
		// if (u_p_cnt > 1000) { std::cout.flush(); polyscope::unshow(); }
	};

	polyscope::show();

	threads_done = true;
	for (auto& thread : threads) thread.join();
}
#endif

























#if 0
// code to visualize Thomson's solution

int main(int argc, char *argv[]) {
	// ImGui is easiest with float/int instead of double/size_t
	float lambda = 1.0, mu = 1.0;
	float force_coeffs[3]{0.0, 1.0, 0.0};
	int num_samples = 5000;
	float sample_radius = 0.5;

	polyscope::init();
	polyscope::options::automaticallyComputeSceneExtents = false;
	/*polyscope::options::warnForInvalidValues = false;*/

	auto compute = [&]() {
		LinearElasticity le{lambda, mu};
		Eigen::Vector3d force; force << force_coeffs[0], force_coeffs[1], force_coeffs[2];
		Eigen::MatrixX3d points(num_samples, 3);
		Eigen::MatrixX3d u_points(num_samples, 3);
		for (size_t i = 0; i < num_samples; ++i) {
			Eigen::Vector3d p = sample_radius * Random::unitBall();
			points.row(i) = p.transpose();
			Eigen::Matrix3d G_p = le.G(p);
			u_points.row(i) = (G_p * force).transpose();
		}
		polyscope::registerPointCloud("sample points", points)
			->setPointRadius(0.0)
			->setPointRenderMode(polyscope::PointRenderMode::Quad)
			->addVectorQuantity("u*", u_points)
				->setVectorRadius(0.1, false)
				->setVectorLengthScale(0.1, false)
				->setEnabled(true);
	};

	compute();

	polyscope::state::userCallback = [&]() {
		/*
		Although the shear modulus, μ, must be positive,
		the Lamé's first parameter, λ, can be negative, in principle;
		however, for most materials it is also positive.
		*/
		ImGui::InputFloat("Lamé parameter lambda", &lambda);
		ImGui::InputFloat("Lamé parameter mu", &mu);

		ImGui::SliderFloat3("point force", force_coeffs, -10.0, 10.0);

		ImGui::SliderInt("number of samples", &num_samples, 1, 10000);
		ImGui::SliderFloat("radius of sample sphere", &sample_radius, 0.0, 1.0);

		if (ImGui::Button("compute")) compute();
	};

	polyscope::show();
}
#endif











#if 0
// Laplacian Walk on Spheres 3D slice

bool done = false; // for multithreading
constexpr int N = 400; // subdivide screen into N*N pixels, run WoS on those points

namespace Util {
	Eigen::Matrix4d glm2eigen(glm::mat4 inp) {
		Eigen::Matrix4d res;
		for (int col = 0; col < 4; ++col) {
			res.col(col) << inp[col][0], inp[col][1], inp[col][2], inp[col][3];
		}
		return res;
	}
	Eigen::Vector3d glm2eigen(glm::vec3 inp) {
		Eigen::Vector3d res;
		res << inp[0], inp[1], inp[2];
		return res;
	}
	double mul(double a, double b) { return a*b; }
	Eigen::Vector3d mul(Eigen::Vector3d& a, Eigen::Vector3d& b) { return a.cwiseProduct(b); }
	template<class T> struct MeanVar {
		// Welford's online algorithm
		// https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford%27s_online_algorithm
		int count{};
		T mean{};
		T m2{};
		void update(T newval) {
			++count;
			T delta = newval - mean;
			mean += delta / count;
			T delta2 = newval - mean;
			m2 += mul(delta, delta2);
		}
		T getMean() { return mean; }
		T getVar() { return m2 / (count - 1); }
		int getCount() { return count; }
	};

}


namespace Mesh {
	Eigen::MatrixX3d V, FN, VN, EN;
	Eigen::MatrixX3i F;
	Eigen::MatrixX2i E;
	Eigen::VectorXi EMAP;
	igl::AABB<Eigen::MatrixX3d, 3> tree;
	igl::WindingNumberAABB<Eigen::RowVector3d, Eigen::MatrixX3d, Eigen::MatrixX3i> hier;
	igl::FastWindingNumberBVH fwn_bvh;
	polyscope::SurfaceMesh* psMesh{};
	polyscope::SurfaceVertexScalarQuantity* u_b{};

	double boundaryValue(Eigen::Vector3d x) {
		// todo
		// return x(0)*x(1)*x(2);
		return ((x(0)>0.0)^(x(1)>0.0)^(x(2)>0.0))-0.5;
	}

	double sourceValue(Eigen::Vector3d x) {
		double norm = x.norm();
		return std::exp(-norm*norm);
	}

	void init() {
		static bool is_initialized = false; assert(!is_initialized); is_initialized = true;
		igl::readOFF(TUTORIAL_SHARED_PATH "/cow.off", V, F);
		tree.init(V, F);
		hier.set_mesh(V, F);
		igl::per_face_normals(V, F, FN);
		igl::per_vertex_normals(V, F, igl::PER_VERTEX_NORMALS_WEIGHTING_TYPE_ANGLE, FN, VN);
		igl::per_edge_normals(V, F, igl::PER_EDGE_NORMALS_WEIGHTING_TYPE_UNIFORM, FN, EN, E, EMAP);
		igl::fast_winding_number(V, F, 2/*Taylor series expansion order to use (e.g., 2)*/, fwn_bvh);
		psMesh = polyscope::registerSurfaceMesh("mesh", V, F);
		std::vector<double> ub(V.rows());
		for (int i = 0; i < V.rows(); ++i) ub[i] = boundaryValue(V.row(i).transpose());
		u_b = psMesh->addVertexScalarQuantity("u_b", ub);
		u_b->setEnabled(true);
	}

	double sdf(Eigen::Vector3d p) {
		// + for outside, - for inside
		Eigen::RowVector3d q = p.transpose();
#if 1
		return igl::signed_distance_winding_number(tree, V, F, hier, q);
#elif 1
		return igl::signed_distance_fast_winding_number(q, V, F, tree, fwn_bvh);
#elif 1
		return igl::signed_distance_pseudonormal(tree, V, F, FN, VN, EN, EMAP, q);
#endif
	}

	std::vector<bool> isInside(const Eigen::MatrixX3d& points) {
		std::vector<bool> res(points.size());
#if 1
		// winding_number / fast_winding_number
		Eigen::VectorXd winding_numbers(points.rows());
		igl::winding_number(V, F, points, winding_numbers);
		// igl::fast_winding_number(fwn_bvh, 2/*parameter controlling accuracy (e.g., 2)*/, points, winding_numbers);
		for (int i = 0; i < points.rows(); ++i)
			res[i] = std::abs(winding_numbers[i]) > 0.5;
#elif 1
		// pseudonormal_test
		Eigen::VectorXi closest_faces;
		Eigen::MatrixX3d closest_points;
		{
			Eigen::VectorXd squared_distances;
			tree.squared_distance(V, F, points, squared_distances, closest_faces, closest_points);
		}
		for (int i = 0; i < points.rows(); ++i) {
			double sign;
			Eigen::RowVector3d closest_point = closest_points.row(i), normal;
			igl::pseudonormal_test(
				V, F, FN, VN, EN, EMAP,
				points.row(i), closest_faces(i), closest_point,
				sign, normal);
			// sign is -1 if inside, +1 if outside
			res[i] = sign < -0.5;
		}
#endif
		return res;
	}
}

namespace SlicePlane {
	polyscope::SlicePlane* psPlane{};

	void init() {
		static bool is_initialized = false; assert(!is_initialized); is_initialized = true;
		psPlane = polyscope::addSceneSlicePlane();
		psPlane->setDrawWidget(true);
	}

	Eigen::Matrix4d getTransform() {
		return Util::glm2eigen(psPlane->getTransform());
	}

	Eigen::Vector3d getNormal() {
		return getTransform().col(0).head<3>();
	}

	double getDisplacement() {
		// not sure if this should be negative or what
		return getNormal().dot(getTransform().col(3).head<3>());
	}
}

namespace Camera {
	Eigen::Matrix4d getView() { return Util::glm2eigen(polyscope::view::getCameraViewMatrix()); }
	Eigen::Matrix4d getMvp() { return Util::glm2eigen(polyscope::view::getCameraPerspectiveMatrix()) * getView(); }
	Eigen::Vector3d getPosition() { return Util::glm2eigen(polyscope::view::getCameraWorldPosition()); }
}

namespace QueryPoints {
	// y*N+x
	std::vector<int> pixelIdxs;
	std::vector<Eigen::Vector3d> points;
	std::vector<Util::MeanVar<double>> u(N * N);
	std::vector<Util::MeanVar<Eigen::Vector3d>> grad_u(N * N);
	std::vector<double> depthVals(N * N, std::numeric_limits<double>::infinity());
	polyscope::PointCloud* psPointCloud{};
	std::mutex mutex;

	void update() {
		Eigen::MatrixX3d unprojected_points(N*N, 3);
		{
			Eigen::Matrix4d mvp = Camera::getMvp();
			Eigen::Vector4d viewport; viewport << 0.0, 0.0, 1.0, 1.0;
			Eigen::Vector4d planeVec; planeVec << SlicePlane::getNormal(), -SlicePlane::getDisplacement();
			for (int y=0; y<N; ++y) for (int x=0; x<N; ++x) {
				Eigen::Vector2d uv; uv << (x+0.5)/N, (y+0.5)/N;
				Eigen::Vector3d unprojected_point;
				igl::unproject_on_plane(uv, mvp, viewport, planeVec, unprojected_point);
				unprojected_points.row(y*N+x) = unprojected_point.transpose();
			}
		}
		std::vector<bool> isInside = Mesh::isInside(unprojected_points);
		{
			std::lock_guard{mutex};

			// clear variables
			std::fill(depthVals.begin(), depthVals.end(), std::numeric_limits<double>::infinity());
			std::fill(u.begin(), u.end(), Util::MeanVar<double>{});
			std::fill(grad_u.begin(), grad_u.end(), Util::MeanVar<Eigen::Vector3d>{});
			pixelIdxs.clear(); points.clear();

			Eigen::Vector3d cameraPosition = Camera::getPosition();
			for (int i = 0; i < N*N; ++i) {
				if (!isInside[i]) continue;
				pixelIdxs.push_back(i);
				points.push_back(unprojected_points.row(i).transpose());
				depthVals[i] = (points.back() - cameraPosition).norm() - 0.1;
			}
		}
		if (psPointCloud) { polyscope::removePointCloud("query points"); psPointCloud = nullptr; }
		assert(!psPointCloud);
		psPointCloud = polyscope::registerPointCloud("query points", points);
		psPointCloud->setIgnoreSlicePlane(SlicePlane::psPlane->name, true);
		//psPointCloud->setPointRenderMode(polyscope::PointRenderMode::Quad);
#if 0
		std::vector<double> sdfs(points.size());
		for(int i=0;i<sdfs.size();++i){
			if(sdfs[i]<0.0)std::cerr<<"SDF NEGATIVE QUERYPOINT ";
			sdfs[i]=-Mesh::sdf(points[i]);
		}
		auto sq=psPointCloud->addScalarQuantity("sdf", sdfs);
		psPointCloud->setPointRadiusQuantity(sq,false);
#endif
	}
}

namespace Walk {
	void walkOnSpheres(int ptIdx) {
		Util::MeanVar<double>& u = QueryPoints::u[ptIdx];
		Util::MeanVar<Eigen::Vector3d>& grad_u = QueryPoints::grad_u[ptIdx];
		const Eigen::Vector3d& x0 = QueryPoints::points[ptIdx];
		Eigen::Vector3d x1{};
		Eigen::Vector3d v_x1{};
		double R0{};

		std::vector<Eigen::Vector3d> xvec, dispvec; std::vector<double>rvec;
		double u_x0{}; {
			// Ri is the distance from xi to xiplus1
			Eigen::Vector3d xi = x0;
			int i=0;
			while (true) {
				// the main algorithm
				double Ri = -Mesh::sdf(xi);
				if (Ri < -1e-4)
					std::cerr << "why is the sdf negative? i="<<i<<", ri="<<Ri<<", xi="<<xi<<". ";
				const Eigen::Vector3d randomUnitVector = Random::unitSphere();
				const Eigen::Vector3d xiplus1 = xi + Ri * randomUnitVector;
				if (i == 0) {
					R0 = Ri;
					v_x1 = randomUnitVector;
					x1 = xiplus1;
				}

				// take the boundary value if we are close
				if (Ri < 1e-3) break;

				// set up the next iteration
				xi = xiplus1;
				++i;
			}
			// MCGP eq 5
			u_x0 = Mesh::boundaryValue(xi);
		}

		// MCGP 2.2.1
		double hat_u = u_x0;

		// MCGP 3.1
		double u_x1 = u_x0;
		Eigen::Vector3d hat_grad_u = 3/R0 * u_x1 * v_x1;

		// control variates

		// MCGP 4.1.1
		double linear_approx_u = grad_u.getMean().dot(x1 - x0);
		double ctl_var_hat_u = hat_u - linear_approx_u;

		// MCGP 4.1.2
		Eigen::Vector3d ctl_var_hat_grad_u = 3/R0 * (hat_u - u.getMean()) * v_x1;

		// u.update(hat_u); grad_u.update(hat_grad_u);
		u.update(ctl_var_hat_u); grad_u.update(ctl_var_hat_grad_u);
	}
	void callback() {
		while (!done) {
			std::lock_guard{QueryPoints::mutex};
			if (QueryPoints::points.empty()) continue;
			std::uniform_int_distribution dist{0, (int)QueryPoints::points.size() - 1};
			int ptIdx = dist(Random::gen);

			walkOnSpheres(ptIdx);
		}
	}
}

namespace RenderImage {
	polyscope::ScalarRenderImageQuantity* psRenderImage{};
	void update() {
		if (psRenderImage && !psRenderImage->isEnabled()) return;
		std::vector<double> mean_u(N * N);
		{
			std::lock_guard{QueryPoints::mutex};
			for (int i = 0; i < QueryPoints::points.size(); ++i) {
				int pixel = QueryPoints::pixelIdxs[i];
				mean_u[pixel] = QueryPoints::u[i].getMean();
				// mean_u[pixel] = QueryPoints::u[i].getVar(); // visualize variance
			}
		}
		if (psRenderImage) { polyscope::removeFloatingQuantity("u"); psRenderImage = nullptr; }
		assert(!psRenderImage);
		psRenderImage = polyscope::addScalarRenderImageQuantity(
				"u", N, N,
				QueryPoints::depthVals,
				std::vector<Eigen::Vector3d>(N*N, -SlicePlane::getNormal()),
				mean_u,
				polyscope::ImageOrigin::LowerLeft);
		psRenderImage->setMapRange(Mesh::u_b->getMapRange());
	}
}

int main(int argc, char *argv[]) {
	polyscope::init();
	polyscope::options::automaticallyComputeSceneExtents = false;
	polyscope::options::warnForInvalidValues = false;
	Mesh::init();
	SlicePlane::init();
	Eigen::Matrix4d planeTransform = SlicePlane::getTransform(), cameraMvp = Camera::getMvp();

	std::thread walkThread(Walk::callback);

	auto next_render_time = std::chrono::system_clock::now();
	polyscope::state::userCallback = [&]() {
		Eigen::Matrix4d newPlaneTransform = SlicePlane::getTransform(), newCameraMvp = Camera::getMvp();
		if (newPlaneTransform != planeTransform || newCameraMvp != cameraMvp) {
			std::lock_guard{QueryPoints::mutex};
			QueryPoints::update();
			planeTransform = newPlaneTransform;
			cameraMvp = newCameraMvp;
		}
		auto now = std::chrono::system_clock::now();
		if (now > next_render_time) {
			RenderImage::update();
			next_render_time = std::chrono::system_clock::now() + std::chrono::milliseconds(200);
		}
	};
	polyscope::show();
	done = true;
	walkThread.join();
}
#endif














#if 0
// polyscope slice planes

#if 0
struct TorusSDF {
	glm::vec2 t;
	TorusSDF(float major_axis, float minor_axis): t{major_axis, minor_axis} {}
	float operator()(glm::vec3 p) {
		float scale = 0.5;
		p /= scale;
		p += glm::vec3{1., 0., 1.};
		glm::vec2 pxz{p.x, p.z};
		glm::vec2 q = glm::vec2(glm::length(pxz) - t.x, p.y);
		return (glm::length(q) - t.y) * scale;
	}
};

struct BoxFrameSDF {
	glm::vec3 b;
	BoxFrameSDF(float x, float y, float z): b{0.5*x,0.5*y,0.5*z} {}
	float operator()(glm::vec3 p) {
		float scale = 0.5;
		p /= scale;
		float e = 0.1;
		p = glm::abs(p) - b;
		glm::vec3 q = glm::abs(p + e) - e;

		float out = glm::min(
				glm::min(
					glm::length(glm::max(glm::vec3(p.x, q.y, q.z), 0.0f)) + glm::min(glm::max(p.x, glm::max(q.y, q.z)), 0.0f),
					glm::length(glm::max(glm::vec3(q.x, p.y, q.z), 0.0f)) + glm::min(glm::max(q.x, glm::max(p.y, q.z)), 0.0f)),
				glm::length(glm::max(glm::vec3(q.x, q.y, p.z), 0.0f)) + glm::min(glm::max(q.x, glm::max(q.y, p.z)), 0.0f));
		return out * scale;
	}
};

template<class A, class B>
struct UnionSDF {
	A a; B b;
	UnionSDF(A a, B b): a{a}, b{b} {}
	float operator()(glm::vec3 p) { return std::min(a(p), b(p)); }
};

template<class A, class B>
struct IntersectionSDF {
	A a; B b;
	IntersectionSDF(A a, B b): a{a}, b{b} {}
	float operator()(glm::vec3 p) { return std::max(a(p), b(p)); }
};

struct SlicePlaneSDF {
	polyscope::SlicePlane* psPlane;
	SlicePlaneSDF(polyscope::SlicePlane* psPlane): psPlane{psPlane} {}
	float operator()(glm::vec3 p) {
		glm::mat4 transform = psPlane->getTransform();
		glm::vec3 n = transform[0];
		float h = glm::dot(n, glm::vec3(transform[3]));
		return std::abs(glm::dot(p, n) - h) - 1e-5;
	}
};


// TODO convert to functions not lambdas
auto colorFunc = [](glm::vec3 p) {
	glm::vec3 color{0., 0., 0.};
	if (p.x > 0) {
		color += glm::vec3{1.0, 0.0, 0.0};
	}
	if (p.y > 0) {
		color += glm::vec3{0.0, 1.0, 0.0};
	}
	if (p.z > 0) {
		color += glm::vec3{0.0, 0.0, 1.0};
	}
	return color;
};

auto scalarFunc = [](glm::vec3 p) { return p.x; };

#endif

struct MeshSDF {
	Eigen::MatrixXd V, FN, VN, EN;
	Eigen::MatrixXi F, E;
	Eigen::VectorXi EMAP;
	igl::AABB<Eigen::MatrixXd,3> tree;

	MeshSDF(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F): V{V}, F{F} {
		// Pseudonormal setup...
		// Precompute signed distance AABB tree
		tree.init(V,F);
		// Precompute vertex,edge and face normals
		igl::per_face_normals(V,F,FN);
		igl::per_vertex_normals(V,F,igl::PER_VERTEX_NORMALS_WEIGHTING_TYPE_ANGLE,FN,VN);
		igl::per_edge_normals(V,F,igl::PER_EDGE_NORMALS_WEIGHTING_TYPE_UNIFORM,FN,EN,E,EMAP);
	}
	float operator()(Eigen::Vector3d p) {
		return igl::signed_distance_pseudonormal(tree,V,F,FN,VN,EN,EMAP,p.transpose());
	}
};





#if 1
void meshPlaneIntersection(
		const Eigen::MatrixX3d& worldV, const Eigen::MatrixX3i& F, const Eigen::MatrixX2i& E, const Eigen::MatrixX3i& FasE,
		// FasE = face id -> three edge ids (could not find in libigl?)
		polyscope::SlicePlane* psPlane, // should be const but for some reason doesn't work
		Eigen::MatrixX3d &nodes, Eigen::MatrixX2d &plane_nodes, Eigen::MatrixX2i &edges
		) {

	glm::mat4 glmPlane2world = psPlane->getTransform(); // plane repr to world repr
	Eigen::Matrix4d plane2world;
	for (int col=0;col<4;++col){
		plane2world.col(col) <<
			glmPlane2world[col][0],
			glmPlane2world[col][1],
			glmPlane2world[col][2],
			glmPlane2world[col][3];}
	Eigen::Matrix4d world2plane = plane2world.inverse();

	Eigen::MatrixX3d planeV(worldV.rows(), worldV.cols());
	for (int row = 0; row < worldV.rows(); ++row) {
		Eigen::Vector4d planev = world2plane * (Eigen::Vector4d() << worldV.row(row).transpose(), 1.0).finished();
		planeV.block<1,3>(row,0) = planev.head<3>();
	}

	std::vector<bool> edgeIsInPlane(E.rows());
	// output will be intersection of mesh (worldV,F) and plane specified by psPlane
	// will be a curve specified by nodes (y,z) and edges (pair of node idxs)
	std::vector<int> edge2node(E.rows(), -1);
	//std::vector<Eigen::Triplet> node_triplets, edge_triplets;
	std::vector<Eigen::Vector3d> node_rows; node_rows.reserve(E.rows());
	std::vector<Eigen::Vector2d> plane_node_rows; plane_node_rows.reserve(E.rows());
	int node_idx = 0;
	for (int e = 0; e < E.rows(); ++e) {
		int v0 = E(e, 0), v1 = E(e, 1);
		double x0 = planeV(v0,0), x1 = planeV(v1,0);
		if (x0*x1 > 0) continue;
		x0 = std::abs(x0); x1 = std::abs(x1);
		double y = (planeV(v0,1)*x1 + planeV(v1,1)*x0) / (x0+x1);
		double z = (planeV(v0,2)*x1 + planeV(v1,2)*x0) / (x0+x1);
		edge2node[e] = node_idx;
		plane_node_rows.push_back((Eigen::Vector2d() << y, z).finished());
		Eigen::Vector4d worldv = plane2world * (Eigen::Vector4d() << 0.0, y, z, 1.0).finished();
		node_rows.push_back(worldv.head<3>());
		++node_idx;
	}
	std::vector<std::vector<int>> edge_adj(E.rows());
	for (int f=0; f<F.rows();++f) {
		int e0 = FasE(f, 0), e1 = FasE(f, 1), e2 = FasE(f, 2);
		if (edge2node[e0]!=-1 || edge2node[e1]!=-1 || edge2node[e2]!=-1) {
			std::array<int,3> nodes{edge2node[e0], edge2node[e1], edge2node[e2]};
			if(nodes[0]==-1)std::swap(nodes[0], nodes[2]);
			else if(nodes[1]==-1)std::swap(nodes[1],nodes[2]);
			assert(nodes[0]!=-1);
			assert(nodes[1]!=-1);
			assert(nodes[2]==-1);
			edge_adj[nodes[0]].emplace_back(nodes[1]);
			edge_adj[nodes[1]].emplace_back(nodes[0]);
		}
	}
	std::vector<Eigen::Vector2i> edge_rows;
	std::vector<bool>vis(E.rows(), false);
	// std::vector<std::vector<int>> loops; // see igl::boundary_loop
	for(int i=0;i<E.rows();++i){
		if(vis[i])continue;
		vis[i]=true;
		if(edge_adj[i].empty())continue;
		// loops.emplace_back();
		// std::vector<int> &loop = loops.back();
		std::vector<int> loop;
		int prev=-1, cur=i;
		double area2 = 0.0;
		do {
			assert(edge_adj[cur].size() == 2);
			loop.push_back(cur);
			int next = edge_adj[cur][0] == prev ? edge_adj[cur][1] : edge_adj[cur][0];
			Eigen::Vector2d x = plane_node_rows[cur], y = plane_node_rows[next];
			area2 += x(0)*y(1) - x(1)*y(0);
			prev = cur; cur = next;
			vis[cur] = true;
		} while (cur!=i);
		if(area2<0.0) {
			// enforce orientation in plane
			std::reverse(begin(loop),end(loop));
		}
		for(int j=0;j<loop.size();++j){
			edge_rows.push_back((Eigen::Vector2i() << loop[j], loop[(j+1)%loop.size()]).finished());
		}
	}
	// map all edges to None | vertex
	// map all faces to None | line between verts
	nodes = Eigen::MatrixX3d(node_rows.size(), 3);
	plane_nodes = Eigen::MatrixX2d(plane_node_rows.size(), 2);
	edges = Eigen::MatrixX2i(edge_rows.size(), 2);
	for(int i=0;i<node_rows.size();++i)
		nodes.row(i)=node_rows[i].transpose();
	for(int i=0;i<plane_node_rows.size();++i)
		plane_nodes.row(i)=plane_node_rows[i].transpose();
	for(int i=0;i<edge_rows.size();++i)
		edges.row(i)=edge_rows[i].transpose();
}
#else
void meshPlaneIntersection2(
		const Eigen::MatrixX3d& V, const Eigen::MatrixX3i& F, 
		polyscope::SlicePlane* psPlane, // should be const but for some reason getTransform is not const
		Eigen::MatrixX3d &nodes, Eigen::MatrixX2d &plane_nodes, Eigen::MatrixX2i &edges
		) {
	glm::mat4 glmPlane2world = psPlane->getTransform(); // plane repr to world repr
	Eigen::Matrix4d plane2world;
	for (int col=0;col<4;++col){
		plane2world.col(col) <<
			glmPlane2world[col][0],
			glmPlane2world[col][1],
			glmPlane2world[col][2],
			glmPlane2world[col][3];}
	Eigen::Matrix4d world2plane = plane2world.eval();
	Eigen::MatrixX3d tetV(4,3); {
		Eigen::RowVector3d x = plane2world.col(0);
		Eigen::RowVector3d y = plane2world.col(1);
		Eigen::RowVector3d z = plane2world.col(2);
		constexpr double sqrt3 = std::sqrt(3);
		tetV << y*2.0 , -y+sqrt3*z , -y-sqrt3*z , x;
		tetV *= 9999.0;
	}
	Eigen::MatrixX3i tetF(4,3); tetF << 0,1,2, 0,1,3, 0,2,3, 1,2,3;
	Eigen::MatrixX3d Vslice; Eigen::MatrixX3i Fslice;
	igl::copyleft::cgal::mesh_boolean(V,F,tetV,tetF,MESH_BOOLEAN_TYPE_INTERSECTION,Vslice,Fslice);
	std::vector<std::vector<int>> loops;
	igl::boundary_loop(Fslice, loops);
	Eigen::MatrixX2d planeV(V.rows(), 2);
	for (int row = 0; row < V.rows(); ++row) {
		Eigen::Matrix4d planev = world2plane * (Eigen::Vector4d() << V.row(row).transpose(), 1.0).finished();
		planeV.row(row) << planev(1), planev(2);
	}
	int num_nodes{};
	for(std::vector<int>&loop:loops){
		num_nodes+=loop.size();
		double area = 0.0;
		for (int i=0;i<loop.size(); ++i) {
			Eigen::RowVector2d x = planeV.row(loop[i]);
			Eigen::RowVector2d y = planeV.row(loop[(i+1)%loop.size()]);
			area += x(0)*y(1) - x(1)*y(0);
		}
		if(area<0.0)std::reverse(loop.begin(),loop.end());
	}
	nodes.resize(num_nodes,3); plane_nodes.resize(num_nodes,2); edges.resize(num_nodes,2);
	int total_i{};
	for(std::vector<int>&loop:loops){
		for (int i=0;i<loop.size(); ++i) {
			plane_nodes.row(total_i) = x;
			nodes.row(total_i) = V.row(loop[i]);
			edges.row(total_i) << loop[i], loop[(i+1)%loop.size()];
		}
	}
}
#endif





template<class T>
struct MeanVar {
	// Welford's online algorithm
	// https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford%27s_online_algorithm
	int count{}; T mean{}, m2{};
	void update(T newval) {
		++count;
		T delta = newval - mean;
		mean += delta / count;
		T delta2 = newval - mean;
		m2 += delta * delta2;
	}
	T getMean() { return mean; }
	T getVar() { return m2 / count; }
};

Eigen::Vector3d randUnit() {
	Eigen::Vector3d res; res << d(gen), d(gen), d(gen);
	res.normalize();
	return res;
}




int main(int argc, char *argv[]) {
	// Initialize polyscope
	polyscope::init();

	//polyscope::options::warnForInvalidValues = false; // nan???
	polyscope::options::automaticallyComputeSceneExtents = false;

	// Read & register the mesh
	Eigen::MatrixX3d meshV;
	Eigen::MatrixX3i meshF;
	igl::readOFF(TUTORIAL_SHARED_PATH "/cow.off", meshV, meshF);
	Eigen::MatrixX2i meshE;
	igl::edges(meshF, meshE);
	Eigen::MatrixX3i meshFasE(meshF.rows(), 3); {
		// face id -> three edge ids (could not find in libigl?)
		std::map<std::array<int,2>, int> vertsToEdge;
		for (int e = 0; e < meshE.rows(); ++e) {
			std::array<int, 2> v{meshE(e, 0), meshE(e, 1)};
			// enforce edge = (small vertex id, big vertex id)
			if (v[0] > v[1]) std::swap(v[0], v[1]);
			vertsToEdge[v] = e;
		}
		for (int f = 0; f < meshF.rows(); ++f) {
			for (int i = 0; i < 3; ++i) {
				std::array<int, 2> v{meshF(f, i), meshF(f, (i+1)%3)};
				if (v[0] > v[1]) std::swap(v[0], v[1]);
				meshFasE(f,i) = vertsToEdge[v];
			}
		}
	}
	MeshSDF meshSDF(meshV,meshF);
	polyscope::SurfaceMesh* psMesh = polyscope::registerSurfaceMesh("input mesh", meshV, meshF);

	// Add a slice plane
	polyscope::SlicePlane* psPlane = polyscope::addSceneSlicePlane();
	psPlane->setDrawWidget(true);


	// show mesh-sliceplane intersection
	// nodes in world space, nodes in plane space, edges
	Eigen::MatrixX3d nodes; Eigen::MatrixX2d plane_nodes; Eigen::MatrixX2i edges;
	polyscope::CurveNetwork* sliceBnd{};
	glm::mat4 sliceBndIsBasedOnPlaneMat(0.0);
	auto updateSliceBnd = [&](){
		if(sliceBnd && !sliceBnd->isEnabled())return;
		// update only when the slice changes
		auto planeMat = psPlane->getTransform();
		if (planeMat == sliceBndIsBasedOnPlaneMat) return;
		meshPlaneIntersection( meshV, meshF, meshE, meshFasE, psPlane, nodes, plane_nodes, edges );
		polyscope::removeCurveNetwork("slice boundary");
		sliceBnd = polyscope::registerCurveNetwork("slice boundary", nodes, edges);
		sliceBnd->setIgnoreSlicePlane(psPlane->name, true);
		sliceBnd->setRadius(1e-3);
		sliceBndIsBasedOnPlaneMat = planeMat;
	};


	constexpr int N = 100; // subdivide screen into N*N points, run WoS for the ones inside slice

	// show the section of the plane that we are looking at
	polyscope::PointCloud* sliceInteriorCloud{};
	glm::mat4 sliceInteriorCloudIsBasedOnPlaneMat(0.0);
	glm::mat4 sliceInteriorCloudIsBasedOnMvp(0.0);
	std::vector<Eigen::Vector3d> sliceInterior; // run WoS on these points
	std::vector<int> pixelInds;
	std::vector<float> depthVals(N*N);
	std::vector<MeanVar<float>> scalarVals;
	std::mutex interiorMutex;
	auto updateSliceInterior = [&](bool force) {
		if(!force&&sliceInteriorCloud&&!sliceInteriorCloud->isEnabled())return;
		// update only when the slice changes or the camera changes
		glm::mat4 planeMat = psPlane->getTransform(); // plane repr to world repr
													  //glm::mat4 cameraMat = polyscope::view::getCameraViewMatrix();
		glm::mat4 view = polyscope::view::getCameraViewMatrix();
		glm::mat4 mvp = polyscope::view::getCameraPerspectiveMatrix() * view;
		if (planeMat == sliceInteriorCloudIsBasedOnPlaneMat && mvp==sliceInteriorCloudIsBasedOnMvp)
			return;
		Eigen::Matrix4d VIEW;for(int i=0;i<4;++i)VIEW.col(i)<<view[i][0],view[i][1],view[i][2],view[i][3];
		Eigen::Matrix4d M; for(int i=0;i<4;++i)M.col(i)<<mvp[i][0],mvp[i][1],mvp[i][2],mvp[i][3];
		Eigen::Vector4d VP; VP << 0.0, 0.0, 1.0, 1.0;
		// glm::vec3 center{objectTransform.get()[3][0], objectTransform.get()[3][1], objectTransform.get()[3][2]};
		// glm::vec3 normal{objectTransform.get()[0][0], objectTransform.get()[0][1], objectTransform.get()[0][2]};
		Eigen::Vector4d P; P << planeMat[0][0],
		planeMat[0][1],
		planeMat[0][2],
		-glm::dot(glm::vec3(planeMat[0]), glm::vec3(planeMat[3]));
		auto unproj = [&](double u, double v) -> Eigen::Vector3d {
			Eigen::Vector2d UV; UV << u, v;
			Eigen::Vector3d Z;
			igl::unproject_on_plane(UV, M, VP, P, Z);
			return Z;
		};
		Eigen::Matrix4d world2plane;
		for(int i=0;i<4;++i)world2plane.col(i)<<planeMat[i][0],planeMat[i][1],planeMat[i][2],planeMat[i][3];
		world2plane = world2plane.inverse().eval();
		Eigen::MatrixX3d points(N*N, 3);
		Eigen::MatrixXd plane_points(N*N, 2);
		for(int j=0;j<N;++j)for(int i=0;i<N;++i){
			Eigen::Vector3d pt = unproj((i+0.5)/N, (j+0.5)/N);
			points.row(j*N+i) = pt;
			depthVals[j*N+i] = -(VIEW*(Eigen::Vector4d() << pt , 1.0).finished()).eval()(2);
			Eigen::Vector3d planept = (world2plane*(Eigen::Vector4d() << pt, 1.0).finished()).eval().head<3>();
			assert(std::abs(planept(0))<1e-3);
			plane_points.row(j*N+i) = planept.tail<2>();
		}
		Eigen::VectorXd winding_numbers(N*N);
		igl::winding_number(
				plane_nodes/*MatrixX2d*/,
				edges/*MatrixX2i*/,
				plane_points/*MatrixX2d*/,
				winding_numbers/*VectorXd*/);
		std::lock_guard{interiorMutex};
		sliceInterior.clear(); sliceInterior.reserve(N*N);
		pixelInds.clear(); pixelInds.reserve(N*N);
		//std::cout << winding_numbers.minCoeff() << ", " << winding_numbers.maxCoeff() << std::endl;
		for(int j=0;j<N;++j)for(int i=0;i<N;++i){
			//if(std::abs(winding_numbers(i)) < 1.0-1e-5) continue;
			if (std::abs(winding_numbers(j*N+i)) < 0.5) {
				depthVals[j*N+i] = 1.0 / 0.0;
				continue;
			}
			sliceInterior.push_back(points.row(j*N+i));
			pixelInds.push_back(j*N+i);
		}
		scalarVals.clear(); scalarVals.resize(pixelInds.size());
		polyscope::removePointCloud("slice interior");
		std::cout << "redoing slice interior cloud" << std::endl;
		sliceInteriorCloud = polyscope::registerPointCloud("slice interior", sliceInterior);
		//sliceInteriorCloud->addScalarQuantity("winding number", winding_numbers);
		sliceInteriorCloud->setIgnoreSlicePlane(psPlane->name, true);
		sliceInteriorCloudIsBasedOnPlaneMat = planeMat;
		sliceInteriorCloudIsBasedOnMvp = mvp;//sliceInteriorCloudIsBasedOnCameraMat = cameraMat;
	};


	polyscope::ScalarRenderImageQuantity* renderImage{};
	glm::mat4 renderImageIsBasedOnPlaneMat(0.0);
	glm::mat4 renderImageIsBasedOnMvp(0.0);
	auto next_update = std::chrono::system_clock::now();
	auto updateRenderImage = [&](){
		auto current_time = std::chrono::system_clock::now();
		if(current_time<next_update && renderImage && !renderImage->isEnabled())return;
		// update only when the slice changes or the camera changes
		glm::mat4 planeMat = psPlane->getTransform(); // plane repr to world repr
		glm::mat4 view = polyscope::view::getCameraViewMatrix();
		glm::mat4 mvp = polyscope::view::getCameraPerspectiveMatrix() * view;
		if (current_time<next_update && planeMat == renderImageIsBasedOnPlaneMat && mvp==renderImageIsBasedOnMvp)
			return;
		next_update = current_time + std::chrono::milliseconds(100);
		updateSliceInterior(true);
		Eigen::Vector3d normal; normal << planeMat[0][0],planeMat[0][1],planeMat[0][2];
		std::vector<float> scalars(N*N);
		std::lock_guard{interiorMutex};
		for(int i=0;i<pixelInds.size();++i) {
			if(scalarVals[i].getMean())std::cout << "mean of " << scalarVals[i].getMean() << std::endl;
			scalars[pixelInds[i]] = scalarVals[i].getMean();
		}
		polyscope::removeFloatingQuantity("WoS");
		std::cout << "updating WoS image" << std::endl;
		renderImage = polyscope::addScalarRenderImageQuantity("WoS", N, N,
				depthVals,
				std::vector<Eigen::Vector3d>(N*N,normal),
				scalars,
				polyscope::ImageOrigin::LowerLeft);
		renderImageIsBasedOnPlaneMat = planeMat;
		renderImageIsBasedOnMvp = mvp;
	};

	auto boundaryValue = [&](Eigen::Vector3d x) {
thread_local std::random_device rd{};
thread_local std::mt19937 gen{rd()};
thread_local std::normal_distribution d{};
		// TODO
		return x(0) + x(1);
	};




	polyscope::init();

	polyscope::ImplicitRenderOpts opts;
	// opts.mode = polyscope::ImplicitRenderMode::FixedStep;
	polyscope::ImplicitRenderMode mode = polyscope::ImplicitRenderMode::SphereMarch;
	// polyscope::ImplicitRenderMode mode = polyscope::ImplicitRenderMode::FixedStep;
	opts.subsampleFactor = 2;

#if 1
#elif 0
	auto rerender = [&]() -> polyscope::ColorRenderImageQuantity* {
		std::cout << "removing surface" << std::endl;
		polyscope::removeStructure("surface"); // errorIfAbsent=false
		std::cout << "rendering surface" << std::endl;
		return polyscope::renderImplicitSurfaceColor("surface", slicedMeshSDF, colorFunc, mode, opts);
	};
	polyscope::ColorRenderImageQuantity* img = rerender();
#else
	auto rerender = [&]() -> polyscope::DepthRenderImageQuantity* {
		//std::cout << "removing surface" << std::endl;
		polyscope::removeStructure("surface"); // errorIfAbsent=false
											   //std::cout << "rendering surface" << std::endl;

		/*
		   void igl::unproject_on_plane(UV, M, VP, P, Z)   

		   Given a screen space point (u,v) and the current projection matrix
		   (e.g. gl_proj * gl_modelview) and viewport,
		   unproject the point into the scene so that it lies on given plane.

		   [in]  UV  2-long uv-coordinates of screen space point
		   [in]  M 4 by 4 projection matrix
		   [in]  VP  4-long viewport: (corner_u, corner_v, width, height)
		   [in]  P 4-long plane equation coefficients: P*(X 1) = 0
		   [out] Z 3-long world coordinate 
		 */
		Eigen::Matrix4d M; {
			glm::mat4 mvp = polyscope::view::getCameraPerspectiveMatrix() * polyscope::view::getCameraViewMatrix();
			for (int i = 0; i < 4; ++i) M.col(i) << mvp[i][0], mvp[i][1], mvp[i][2], mvp[i][3];
			/*std::cout << "M =\n" << M << " = M" << std::endl;*/ }
		Eigen::Vector4d VP; VP << 0.0, 0.0, 1.0, 1.0;
		std::array<float, 3> normal;
		Eigen::Vector4d P; {
			glm::mat4 transform = psPlane->getTransform();
			// glm::vec3 center{objectTransform.get()[3][0], objectTransform.get()[3][1], objectTransform.get()[3][2]};
			// glm::vec3 normal{objectTransform.get()[0][0], objectTransform.get()[0][1], objectTransform.get()[0][2]};
			normal[0] = transform[0][0]; normal[1] = transform[0][1]; normal[2] = transform[0][2];
			P << transform[0][0], transform[0][1], transform[0][2], glm::dot(glm::vec3(transform[0]), glm::vec3(transform[3])); }

		constexpr int WIDTH = 400, HEIGHT = 300;


		Eigen::MatrixXd
			std::vector<Eigen::Vector3d> unproject(WIDTH * HEIGHT);
		glm::vec3 camPos = polyscope::view::getCameraWorldPosition();
		for (int xw = 0; xw < WIDTH; ++xw) {
			for (int yh = 0; yh < HEIGHT; ++yh) {
				Eigen::Vector2d UV; UV << (float)xw/WIDTH, (float)yh/HEIGHT;
				igl::unproject_on_plane(UV, M, VP, P, unproject[yh*WIDTH+xw]);
				// Eigen::Vector3d campos; campos << camPos[0], camPos[1], camPos[2];
				// depthVals[yh*WIDTH+xw] = (Z - campos).norm();
			}
		}
		std::vector<float> depthVals(WIDTH * HEIGHT, 0.0);

		return polyscope::addDepthRenderImageQuantity("surface", WIDTH, HEIGHT,
				depthVals,
				std::vector<std::array<float,3>>(WIDTH*HEIGHT,normal),
				polyscope::ImageOrigin::UpperLeft);
	};
	//polyscope::DepthRenderImageQuantity* img = rerender();
#endif

	bool done=false;
	std::vector<std::thread> threads;
	threads.emplace_back( [&] () {
		while(!done){
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
			std::lock_guard{interiorMutex};
			if(pixelInds.empty())continue;
			std::uniform_int_distribution dist{0, (int)pixelInds.size()-1};
			int i = pixelInds[dist(gen)];
			Eigen::Vector3d x = sliceInterior[i];
			double r;
			do {
				r = meshSDF(x);
				x += randUnit() * r;
			} while (r > 1e-3);
			std::cout << "went from " << sliceInterior[i].transpose() << " to " << x.transpose() << ". adding " << boundaryValue(x) << std::endl;
			scalarVals[i].update(boundaryValue(x));
		}
	});
#define SPAWN(stmts) threads.emplace_back([&](){while(!done){ \
		stmts;std::this_thread::sleep_for(std::chrono::milliseconds(100));}})
	//SPAWN(updateSliceBnd();updateSliceInterior(false);updateRenderImage());
	// multithreading does not do well with polyscope

	// Specify the callback
	polyscope::state::userCallback = [&]() {
		updateSliceBnd();
		updateSliceInterior(false);
		updateRenderImage();
	};

	// Show the gui
	polyscope::show();

	done = true;
	for(auto&t:threads)t.join();
}

#endif

















#if 0
// libigl tutorial 702 winding number
#include <igl/barycenter.h>
#include <igl/boundary_facets.h>
#include <igl/parula.h>
#include <igl/readMESH.h>
#include <igl/marching_tets.h>
#include <igl/winding_number.h>
#include <igl/opengl/glfw/Viewer.h>
#include <Eigen/Sparse>
#include <iostream>


Eigen::MatrixXd V,BC;
Eigen::VectorXd W;
Eigen::MatrixXi T,F,G;
double slice_z = 0.5;
enum OverLayType
{
	OVERLAY_NONE = 0,
	OVERLAY_INPUT = 1,
	OVERLAY_OUTPUT = 2,
	NUM_OVERLAY = 3,
} overlay = OVERLAY_NONE;

void update_visualization(igl::opengl::glfw::Viewer & viewer)
{
	using namespace Eigen;
	using namespace std;
	Eigen::Vector4d plane(
			0,0,1,-((1-slice_z)*V.col(2).minCoeff()+slice_z*V.col(2).maxCoeff()));
	MatrixXd V_vis;
	MatrixXi F_vis;
	VectorXi J;
	{
		SparseMatrix<double> bary;
		// Value of plane's implicit function at all vertices
		const VectorXd IV = 
			(V.col(0)*plane(0) + 
			 V.col(1)*plane(1) + 
			 V.col(2)*plane(2)).array()
			+ plane(3);
		igl::marching_tets(V,T,IV,V_vis,F_vis,J,bary);
	}
	VectorXd W_vis = W(J);
	MatrixXd C_vis;
	// color without normalizing
	igl::parula(W_vis,false,C_vis);


	const auto & append_mesh = [&C_vis,&F_vis,&V_vis](
			const Eigen::MatrixXd & V,
			const Eigen::MatrixXi & F,
			const RowVector3d & color)
	{
		F_vis.conservativeResize(F_vis.rows()+F.rows(),3);
		F_vis.bottomRows(F.rows()) = F.array()+V_vis.rows();
		V_vis.conservativeResize(V_vis.rows()+V.rows(),3);
		V_vis.bottomRows(V.rows()) = V;
		C_vis.conservativeResize(C_vis.rows()+F.rows(),3);
		C_vis.bottomRows(F.rows()).rowwise() = color;
	};
	switch(overlay)
	{
		case OVERLAY_INPUT:
			append_mesh(V,F,RowVector3d(1.,0.894,0.227));
			break;
		case OVERLAY_OUTPUT:
			append_mesh(V,G,RowVector3d(0.8,0.8,0.8));
			break;
		default:
			break;
	}
	viewer.data().clear();
	viewer.data().set_mesh(V_vis,F_vis);
	viewer.data().set_colors(C_vis);
	viewer.data().set_face_based(true);
}

bool key_down(igl::opengl::glfw::Viewer& viewer, unsigned char key, int mod)
{
	switch(key)
	{
		default:
			return false;
		case ' ':
			overlay = (OverLayType)((1+(int)overlay)%NUM_OVERLAY);
			break;
		case '.':
			slice_z = std::min(slice_z+0.01,0.99);
			break;
		case ',':
			slice_z = std::max(slice_z-0.01,0.01);
			break;
	}
	update_visualization(viewer);
	return true;
}

int main(int argc, char *argv[])
{
	using namespace Eigen;
	using namespace std;

	cout<<"Usage:"<<endl;
	cout<<"[space]  toggle showing input mesh, output mesh or slice "<<endl;
	cout<<"         through tet-mesh of convex hull."<<endl;
	cout<<"'.'/','  push back/pull forward slicing plane."<<endl;
	cout<<endl;

	// Load mesh: (V,T) tet-mesh of convex hull, F contains facets of input
	// surface mesh _after_ self-intersection resolution
	igl::readMESH(TUTORIAL_SHARED_PATH "/big-sigcat.mesh",V,T,F);

	// Compute barycenters of all tets
	igl::barycenter(V,T,BC);

	// Compute generalized winding number at all barycenters
	cout<<"Computing winding number over all "<<T.rows()<<" tets..."<<endl;
	igl::winding_number(V,F,BC,W);

	// Extract interior tets
	MatrixXi CT((W.array()>0.5).count(),4);
	{
		size_t k = 0;
		for(size_t t = 0;t<T.rows();t++)
		{
			if(W(t)>0.5)
			{
				CT.row(k) = T.row(t);
				k++;
			}
		}
	}
	// find bounary facets of interior tets
	igl::boundary_facets(CT,G);
	// boundary_facets seems to be reversed...
	G = G.rowwise().reverse().eval();

	// normalize
	W = (W.array() - W.minCoeff())/(W.maxCoeff()-W.minCoeff());

	// Plot the generated mesh
	igl::opengl::glfw::Viewer viewer;
	update_visualization(viewer);
	viewer.callback_key_down = &key_down;
	viewer.launch();
}
#endif







#if 0
// libigl tutorial 704 signed distances

#include <igl/cat.h>
#include <igl/edge_lengths.h>
#include <igl/per_edge_normals.h>
#include <igl/per_face_normals.h>
#include <igl/per_vertex_normals.h>
#include <igl/point_mesh_squared_distance.h>
#include <igl/readMESH.h>
#include <igl/signed_distance.h>
#include <igl/slice_mask.h>
#include <igl/marching_tets.h>
#include <igl/upsample.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/writeOBJ.h>
#include <Eigen/Sparse>
#include <iostream>


Eigen::MatrixXd V;
Eigen::MatrixXi T,F;

igl::AABB<Eigen::MatrixXd,3> tree;
igl::FastWindingNumberBVH fwn_bvh;

Eigen::MatrixXd FN,VN,EN;
Eigen::MatrixXi E;
Eigen::VectorXi EMAP;
double max_distance = 1;

double slice_z = 0.5;
bool overlay = false;

bool useFastWindingNumber = false;


const Eigen::MatrixXd CM = 
(Eigen::MatrixXd(50,3)<<
 242,242,242,
 247,251,253,
 228,234,238,
 233,243,249,
 214,227,234,
 217,234,244,
 199,218,230,
 203,226,240,
 186,211,226,
 187,217,236,
 171,203,222,
 173,209,232,
 157,195,218,
 158,201,228,
 142,187,214,
 143,193,223,
 129,179,210,
 128,185,219,
 114,171,206,
 112,176,215,
 100,163,202,
 98,168,211,
 86,156,198,
 82,159,207,
 71,148,194,
 255,247,223,
 242,230,204,
 255,235,206,
 242,219,189,
 255,225,191,
 242,209,175,
 255,214,176,
 242,198,159,
 255,203,160,
 242,188,145,
 255,192,145,
 242,177,129,
 255,181,128,
 242,167,115,
 255,170,113,
 242,157,101,
 255,159,97,
 242,146,85,
 255,148,82,
 242,136,71,
 255,137,65,
 242,125,55,
 255,126,50,
 242,115,41,
 255,116,36).finished()/255.0;

void update_visualization(igl::opengl::glfw::Viewer & viewer)
{
	using namespace Eigen;
	using namespace std;
	Eigen::Vector4d plane(
			0,0,1,-((1-slice_z)*V.col(2).minCoeff()+slice_z*V.col(2).maxCoeff()));
	MatrixXd V_vis;
	MatrixXi F_vis;
	// Extract triangle mesh slice through volume mesh and subdivide nasty
	// triangles
	{
		VectorXi J;
		SparseMatrix<double> bary;
		{
			// Value of plane's implicit function at all vertices
			const VectorXd IV = 
				(V.col(0)*plane(0) + 
				 V.col(1)*plane(1) + 
				 V.col(2)*plane(2)).array()
				+ plane(3);
			igl::marching_tets(V,T,IV,V_vis,F_vis,J,bary);
			igl::writeOBJ("vis.obj",V_vis,F_vis);
		}
		while(true)
		{
			MatrixXd l;
			igl::edge_lengths(V_vis,F_vis,l);
			l /= (V_vis.colwise().maxCoeff() - V_vis.colwise().minCoeff()).norm();
			const double max_l = 0.03;
			if(l.maxCoeff()<max_l)
			{
				break;
			}
			Array<bool,Dynamic,1> bad = l.array().rowwise().maxCoeff() > max_l;
			MatrixXi F_vis_bad, F_vis_good;
			igl::slice_mask(F_vis,bad,1,F_vis_bad);
			igl::slice_mask(F_vis,(bad!=true).eval(),1,F_vis_good);
			igl::upsample(V_vis,F_vis_bad);
			F_vis = igl::cat(1,F_vis_bad,F_vis_good);
		}
	}

	// Compute signed distance
	VectorXd S_vis;

	if (!useFastWindingNumber)
	{
		VectorXi I;
		MatrixXd N,C;
		// Bunny is a watertight mesh so use pseudonormal for signing
		signed_distance_pseudonormal(V_vis,V,F,tree,FN,VN,EN,EMAP,S_vis,I,C,N);
	} else {
		signed_distance_fast_winding_number(V_vis, V, F, tree, fwn_bvh, S_vis);
	}    

	const auto & append_mesh = [&F_vis,&V_vis](
			const Eigen::MatrixXd & V,
			const Eigen::MatrixXi & F,
			const RowVector3d & color)
	{
		F_vis.conservativeResize(F_vis.rows()+F.rows(),3);
		F_vis.bottomRows(F.rows()) = F.array()+V_vis.rows();
		V_vis.conservativeResize(V_vis.rows()+V.rows(),3);
		V_vis.bottomRows(V.rows()) = V;
	};
	if(overlay)
	{
		append_mesh(V,F,RowVector3d(0.8,0.8,0.8));
	}
	viewer.data().clear();
	viewer.data().set_mesh(V_vis,F_vis);
	viewer.data().set_colormap(CM);
	viewer.data().set_data(S_vis);
	viewer.core().lighting_factor = overlay;
}

bool key_down(igl::opengl::glfw::Viewer& viewer, unsigned char key, int mod)
{
	switch(key)
	{
		default:
			return false;
		case ' ':
			overlay ^= true;
			break;
		case '.':
			slice_z = std::min(slice_z+0.01,0.99);
			break;
		case ',':
			slice_z = std::max(slice_z-0.01,0.01);
			break;
		case '1':
			useFastWindingNumber = true;
			break;
		case '2':
			useFastWindingNumber = false;
			break;
	}
	update_visualization(viewer);
	return true;
}

int main(int argc, char *argv[])
{
	using namespace Eigen;
	using namespace std;

	cout<<"Usage:"<<endl;
	cout<<"[space]  toggle showing surface."<<endl;
	cout<<"'.'/','  push back/pull forward slicing plane."<<endl;
	cout<< "1/2 toggle between fast winding number (1) and pseudonormal (2) signing. \n";
	cout<<endl;

	// Load mesh: (V,T) tet-mesh of convex hull, F contains original surface
	// triangles
	igl::readMESH(TUTORIAL_SHARED_PATH "/bunny.mesh",V,T,F);


	// Encapsulated call to point_mesh_squared_distance to determine bounds
	{
		VectorXd sqrD;
		VectorXi I;
		MatrixXd C;
		igl::point_mesh_squared_distance(V,V,F,sqrD,I,C);
		max_distance = sqrt(sqrD.maxCoeff());
	}

	// Fast winding and Pseudo normal depend on differnt AABB trees... We initialize both here.

	// Pseudonormal setup...
	// Precompute signed distance AABB tree
	tree.init(V,F);
	// Precompute vertex,edge and face normals
	igl::per_face_normals(V,F,FN);
	igl::per_vertex_normals(
			V,F,igl::PER_VERTEX_NORMALS_WEIGHTING_TYPE_ANGLE,FN,VN);
	igl::per_edge_normals(
			V,F,igl::PER_EDGE_NORMALS_WEIGHTING_TYPE_UNIFORM,FN,EN,E,EMAP);

	// fast winding number setup (just init fwn bvh)
	igl::fast_winding_number(V, F, 2, fwn_bvh);

	// Plot the generated mesh
	igl::opengl::glfw::Viewer viewer;
	update_visualization(viewer);
	viewer.callback_key_down = &key_down;
	viewer.data().show_lines = false;
	viewer.launch();
}


#endif












#if 0
// set up polyscope / debug igl with vouga

#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"
#include <igl/boundary_facets.h>
#include <igl/colon.h>
#include <igl/cotmatrix.h>
#include <igl/jet.h>
#include <igl/min_quad_with_fixed.h>
#include <igl/readOFF.h>
#include <igl/setdiff.h>
#include <igl/slice.h>
#include <igl/unique.h>
#include <igl/opengl/glfw/Viewer.h>
#include <Eigen/Sparse>
#include <iostream>

int main(int argc, char *argv[])
{
	using namespace Eigen;
	using namespace std;
	MatrixXd V;
	MatrixXi F;
	igl::readOFF(argv[1], V, F);
	// Find boundary edges
	MatrixXi E;
	igl::boundary_facets(F,E);
	// Find boundary vertices
	VectorXi b,IA,IC;
	// unique(A, C, IA, IC) If A is a vector, then C = A(ia) and A = C(ic)
	igl::unique(E,b,IA,IC);
	// List of all vertex indices
	VectorXi all,in;
	// colon(low, hi, I) Enumerates values between low and hi with unit step.
	igl::colon<int>(0,V.rows()-1,all);
	// List of interior indices
	// setdiff(A,B,C,IA) C=unique elements in A but not B, IA=indices such that C=A(IA)
	igl::setdiff(all,b,in,IA);

	// Construct and slice up Laplacian
	SparseMatrix<double> L,L_in_in,L_in_b;
	igl::cotmatrix(V,F,L);
	// slice(X,R,C,Y) like matlab Y=X(R,C)
	igl::slice(L,in,in,L_in_in);
	igl::slice(L,in,b,L_in_b);

	// Dirichlet boundary conditions from z-coordinate
	VectorXd Z = V.col(2);
	//VectorXd::Constant(V.rows(),1,1.0);
	VectorXd bc = Z(b);

	std::set<int> bnd_inds;


	VectorXd bnd = VectorXd::Zero(V.rows(),1);
	int num_bnd =0 ;//, num_in = 0;
	for(int i:b){
		bnd[i]=1.0;
		bnd_inds.insert(i);
		num_bnd++;
	}
	int num_verts=V.rows();
	int cnt=0,cnt2=0;
	typedef Eigen::Triplet<double> T;
	std::vector<T> triplets,triplets_int;
	for(int i=0;i<num_verts;++i){
		if (bnd_inds.count(i)) {
			triplets.emplace_back(cnt,i,1.0);
			cnt++;
		}else{
			triplets_int.emplace_back(cnt2,i,1.0);
			cnt2++;
		}
	}
	SparseMatrix<double> select_bnd(cnt,num_verts);
	select_bnd.setFromTriplets(begin(triplets),end(triplets));
	SparseMatrix<double> select_int(cnt2,num_verts);
	select_int.setFromTriplets(begin(triplets_int),end(triplets_int));
	VectorXd  ones = VectorXd::Ones(num_bnd,1);
	VectorXd bnd2 = select_bnd.transpose()*ones;
	VectorXd interior=select_int.transpose()*VectorXd::Ones(num_verts-num_bnd,1);
	SparseMatrix<double> Cii = select_int*L*select_int.transpose();
	SparseMatrix<double> Cib = select_int*L*select_bnd.transpose();


#if 1
	// Solve PDE
	SimplicialLLT<SparseMatrix<double > > solver(-Cii);
	assert(solver.info() == Eigen::Success);
	// slice into solution
	Eigen::VectorXd bc2 = select_bnd*V.col(2);
	//Z(in) = solver.solve(Cib*bc2);
	Eigen::VectorXd result = select_int.transpose() * solver.solve(Cib*bc2);
	result += select_bnd.transpose()*bc2;
	assert(solver.info() == Eigen::Success);
#endif

#if 0
	// Solve PDE
	SimplicialLLT<SparseMatrix<double > > solver(-L_in_in);
	assert(solver.info() == Eigen::Success);
	// slice into solution
	Z(in) = solver.solve(L_in_b*bc);
	assert(solver.info() == Eigen::Success);
#endif

#if 0
	// Alternative, short hand
	igl::min_quad_with_fixed_data<double> mqwf;
	// Linear term is 0
	VectorXd B = VectorXd::Zero(V.rows(),1);
	// Empty constraints
	VectorXd Beq;
	SparseMatrix<double> Aeq;
	// Our cotmatrix is _negative_ definite, so flip sign
	igl::min_quad_with_fixed_precompute((-L).eval(),b,Aeq,true,mqwf);
	igl::min_quad_with_fixed_solve(mqwf,B,bc,Beq,Z);
#endif

	// Plot the mesh with pseudocolors
#if 0
	igl::opengl::glfw::Viewer viewer;
	viewer.data().set_mesh(V, F);
	viewer.data().show_lines = false;
	viewer.data().set_data(Z);
	std::cout << "Z VECTOR = \n";
	std::cout << Z << std::endl;
	std::cout << "END Z VECTOR\n";
	viewer.launch();
#endif
	// Options
	polyscope::options::autocenterStructures = true;
	polyscope::view::windowWidth = 1024;
	polyscope::view::windowHeight = 1024;

	// Initialize polyscope
	polyscope::init();

	// std::string filename = args::get(inFile);
	// std::cout << "loading: " << filename << std::endl;

	// Read the mesh
	// igl::readOBJ(filename, meshV, meshF);

	// Register the mesh with Polyscope
	auto psmesh = polyscope::registerSurfaceMesh("input mesh", V, F);
	psmesh->addVertexScalarQuantity("Z", Z);
	psmesh->addVertexScalarQuantity("result", result);
	psmesh->addVertexScalarQuantity("bnd", bnd);
	// psmesh->addVertexScalarQuantity("in", in);
	psmesh->addVertexScalarQuantity("bnd2", bnd2);
	psmesh->addVertexScalarQuantity("interior", interior);


	// Add the callback
	// polyscope::state::userCallback = callback;

	// Show the gui
	polyscope::show();
}

#endif






















#if 0
// libigl tutorial 303 laplace equation
#include <igl/boundary_facets.h>
#include <igl/colon.h>
#include <igl/cotmatrix.h>
#include <igl/jet.h>
#include <igl/min_quad_with_fixed.h>
#include <igl/readOFF.h>
#include <igl/setdiff.h>
#include <igl/slice.h>
#include <igl/unique.h>
#include <igl/opengl/glfw/Viewer.h>
#include <Eigen/Sparse>
#include <iostream>

int main(int argc, char *argv[])
{
	using namespace Eigen;
	using namespace std;
	MatrixXd V;
	MatrixXi F;
	igl::readOFF(TUTORIAL_SHARED_PATH "/camelhead.off",V,F);
	// Find boundary edges
	MatrixXi E;
	igl::boundary_facets(F,E);
	// Find boundary vertices
	VectorXi b,IA,IC;
	igl::unique(E,b,IA,IC);
	// List of all vertex indices
	VectorXi all,in;
	igl::colon<int>(0,V.rows()-1,all);
	// List of interior indices
	igl::setdiff(all,b,in,IA);

	// Construct and slice up Laplacian
	SparseMatrix<double> L,L_in_in,L_in_b;
	igl::cotmatrix(V,F,L);
	igl::slice(L,in,in,L_in_in);
	igl::slice(L,in,b,L_in_b);

	// Dirichlet boundary conditions from z-coordinate
	VectorXd Z = V.col(2);
	VectorXd bc = Z(b);

	// Solve PDE
	SimplicialLLT<SparseMatrix<double > > solver(-L_in_in);
	// slice into solution
	Z(in) = solver.solve(L_in_b*bc).eval();

	// Alternative, short hand
	igl::min_quad_with_fixed_data<double> mqwf;
	// Linear term is 0
	VectorXd B = VectorXd::Zero(V.rows(),1);
	// Empty constraints
	VectorXd Beq;
	SparseMatrix<double> Aeq;
	// Our cotmatrix is _negative_ definite, so flip sign
	igl::min_quad_with_fixed_precompute((-L).eval(),b,Aeq,true,mqwf);
	igl::min_quad_with_fixed_solve(mqwf,B,bc,Beq,Z);

	// Plot the mesh with pseudocolors
	igl::opengl::glfw::Viewer viewer;
	viewer.data().set_mesh(V, F);
	viewer.data().show_lines = false;
	viewer.data().set_data(Z);
	viewer.launch();
}
#endif
