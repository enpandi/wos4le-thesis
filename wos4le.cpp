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
				return multiplier * walk_params.BoundaryDisplacement(closest_point);
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
	polyscope::options::warnForInvalidValues = false;

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
