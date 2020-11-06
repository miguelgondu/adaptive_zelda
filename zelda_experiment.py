import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, DotProduct
from scipy.spatial import cKDTree

from utils.pcg import level_from_text
from utils.map_elites import load_df_from_generation

class ZeldaExperiment:
    """
    This class is supposed to be used at EVERY iteration.
    Outside of it we maintain the behaviors and times,
    similar to how we did it in Sudoku.
    """
    def __init__(self, path, goal, behaviors=[], times=[], projection=None, verbose=True, acquisition="ucb"):
        self.path = path
        self.prior = load_df_from_generation(path)
        
        if projection is None:
            projection = ["leniency", "reachability", "space coverage"]
        self.projection = projection
        
        if len(projection) == 2:
            # modifies the prior in-function.
            self._project_to_2D()

        self.goal = goal
        # if the projection is something, we need to slice the behs.
        if len(projection) == 2:
            all_feats = ["leniency", "reachability", "space coverage"]
            index_0 = all_feats.index(projection[0])
            index_1 = all_feats.index(projection[1])
            self.behaviors = [[beh[index_0], beh[index_1]] for beh in behaviors]
        elif len(projection) == 3:
            self.behaviors = behaviors
        else:
            raise ValueError(f"projection should be None, or of length 2 or 3. We got {projection}")

        self.times = times
        self.log_times = [np.log(t) for t in self.times]
        self.verbose = verbose
        self.acquisition = acquisition # either "ucb" or "ei"
        self.kappa = 0.03 # for ucb.

        self.domain = np.array(
            [self.prior.loc[i, projection] for i in self.prior.index]
        )

        # Construct a KDTree here with the points in the domain for
        # querying which cell to update.
        self.domain_tree = cKDTree(self.domain)
        self.indices_tested = [self.domain_tree.query(beh)[1] for beh in self.behaviors]

        self.kernel = (
            1 * RBF(length_scale=[1]*len(projection)) +
            1 * DotProduct() +
            WhiteKernel(noise_level=np.log(2)))
        self.gpr = GaussianProcessRegressor(kernel=self.kernel)

        if len(self.behaviors) == len(self.times) and len(self.times) > 0:
            self._fit_gpr()

    def _project_point(self, b) -> list:
        """
        This function takes a point (l, r, s) and
        projects it to whatever self.projection is
        saying. (Could be (l, s), (r, s), (l, s)).
        """
        assert len(b) > 2
        assert len(self.projection) == 2
        all_features = ["leniency", "reachability", "space coverage"]
        return [
            b[all_features.index(self.projection[0])],
            b[all_features.index(self.projection[1])]
        ]

    def _project_to_2D(self):
        b1, b2 = self.projection
        new_df = self.prior.groupby([b1, b2]).mean()
        levels = []
        for idx in new_df.index:
            # This one contains the groupings' values
            df_slice = self.prior[self.prior[b1] == idx[0]]
            df_slice = df_slice[df_slice[b2] == idx[1]]
            level = df_slice["level"][df_slice["performance"].idxmax]
            levels.append(level)
            
        new_df["level"] = levels
        new_df.reset_index(inplace=True)
        new_df = new_df.loc[:, [b1, b2, "performance", "level"]]
        # TODO: maybe I should do a reset_index(inplace=True)
        # print(f"new prior: {new_df}")
        self.prior = new_df

    def _fit_gpr(self):
        """
        Fits a Gaussian Process on log(t(x)) - prior(x),
        where x is the amount of digits we're giving
        the player.
        """

        X = np.array(self.behaviors.copy())
        Y = []
        for log_time, beh in zip(self.log_times, self.behaviors):
            # print(f"Matching behavior {beh}")
            index = self.domain_tree.query(beh)[1]
            # print(f"With centroid {self.domain[index]} (which should be equal to {self.prior.loc[index, ['b1', 'b2', 'b3']]})")
            Y.append(log_time - self.prior.loc[index, ["performance"]])
        Y = np.array(Y)

        if len(X) > 0 and len(Y) > 0:
            if self.verbose:
                print(f"Fitting the GPR with")
                print(f"X: {X}, Y: {Y}")
            self.gpr.fit(X, Y)

    def next_level(self):
        next_behavior = self._acquisition()
        index = self.domain.tolist().index(list(next_behavior))
        next_level = self.prior.loc[index]["level"]
        its_prior_performance = self.prior.loc[index]["performance"]

        mu, sigma = self._compute_mu_and_sigma()
        if self.verbose:
            print(f"Performance in prior: {np.exp(its_prior_performance)}")
            print(f"Predicted performance: {np.exp(mu[index])}")
        next_level = level_from_text(next_level)
        return next_level

    def _acquisition(self):
        if self.acquisition == "ucb":
            acq = self._ucb(kappa=self.kappa)
        elif self.acquisition == "ei":
            acq = self._expected_improvement()
        else:
            print(f"Unexpected value in self.acquisition: {self.acquisition}")
            print(f"Defaulting to UCB")
            acq = self._ucb(kappa=self.kappa)

        next_behavior = list(self.domain[np.argmax(acq)])
        return next_behavior

    def _expected_improvement(self, return_mu_and_sigma=False):
        """
        Computes the EI by sampling.
        """
        n_samples = 10000

        if len(self.times) > 0:
            max_so_far = max([self._g(log_t) for log_t in self.log_times])
        else:
            # At the beginning, max_so_far should be -infinity, no?
            max_so_far = -9.9e150

        prior = self.prior["performance"].values.reshape(-1, 1)

        mu, sigma = self.gpr.predict(self.domain, return_std=True)
        mu_gp = mu.copy()
        sigma = sigma.reshape(-1, 1)
        mu = prior + mu.reshape(-1, 1)

        random_sigmas = sigma * np.random.randn(mu.shape[0], n_samples)
        mu_samples = mu + random_sigmas
        g_samples = self._g(mu_samples)
        print(np.maximum(0, g_samples - max_so_far))
        ei = np.mean(np.maximum(0, g_samples - max_so_far), axis=1)

        if return_mu_and_sigma:
            return (ei, mu_gp, sigma, g_samples)
        else:
            return ei

    def _compute_mu_and_sigma(self):
        if len(self.times) > 0:
            prior = self.prior["performance"].values.reshape(-1, 1)
        else:
            prior = self.prior["performance"].values

        mu, sigma = self.gpr.predict(self.domain, return_std=True)

        if len(self.times) > 0:
            mu = prior + mu.reshape(-1, 1)
        else:
            mu = prior + mu

        if len(self.times) > 0:
            sigma = sigma.reshape(-1, 1)

        return mu.flatten(), sigma.flatten()

    def _ucb(self, kappa=0.03):
        if len(self.times) > 0:
            prior = self.prior["performance"].values.reshape(-1, 1)
        else:
            prior = self.prior["performance"].values

        mu, sigma = self.gpr.predict(self.domain, return_std=True)
        if len(self.times) > 0:
            mu = prior + mu.reshape(-1, 1)
        else:
            mu = prior + mu

        if len(self.times) > 0:
            sigma = sigma.reshape(-1, 1)

        ucb = mu + kappa * sigma

        return self._g(ucb)

    def _g(self, log_t):
        return - (np.exp(log_t) - self.goal) ** 2

    def plot_projected(self, path):
        assert len(self.projection) == 2
        points = self.domain.copy()
        mu, _ = self._compute_mu_and_sigma()

        black_dot = None
        if len(self.behaviors) > 0:
            black_dot = self.behaviors[-1]

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(3*6, 6))
        fig.suptitle(f"t = {self.times}")

        plot = ax1.scatter(
            points[:, 0],
            points[:, 1],
            c=np.exp(mu),
            marker="s",
            s=20
        )

        if black_dot is not None:
            ax1.plot(black_dot[0], black_dot[1], "ok", markersize=12)

        ax1.set_title("Time steps to solve")
        ax1.set_xlabel(self.projection[0])
        ax1.set_ylabel(self.projection[1])
        plt.colorbar(plot, ax=ax1)

        if self.acquisition == "ucb":
            acq = self._ucb(kappa=self.kappa)
        elif self.acquisition == "ei":
            acq = self._expected_improvement()
        else:
            print(f"Unexpected value in self.acquisition: {self.acquisition}")
            print(f"Defaulting to UCB")
            acq = self._ucb(kappa=self.kappa)

        plot2 = ax2.scatter(
            points[:, 0],
            points[:, 1],
            # c=-np.abs(np.exp(mu)-self.goal),
            c=acq,
            marker="s",
            s=20
        )
        if black_dot is not None:
            ax2.plot(black_dot[0], black_dot[1], "ok", markersize=12)

        ax2.set_title(f"Acquisition Function ({self.acquisition})")
        ax2.set_xlabel(self.projection[0])
        ax2.set_ylabel(self.projection[1])
        plt.colorbar(plot2, ax=ax2)
    #     print(dir(self.gpr))
    #     print(self.gpr.get_params())
        if len(self.times) > 0:
            ax2.set_xlabel(f"{self.gpr.kernel_}")
        else:
            ax2.set_xlabel(f"{self.gpr.kernel}")

        pure_mu, _ = self.gpr.predict(self.domain, return_std=True)
        plot3 = ax3.scatter(
            points[:, 0],
            points[:, 1],
            c=pure_mu.flatten(),
            marker="s",
            s=20
        )
        ax3.set_title("\"impact\" on prior")
        plt.colorbar(plot3, ax=ax3)

        plt.savefig(path)
        plt.close()
    
    def save_3D_plot(self, save_path, plot_sigma=True):
        assert (len(self.projection) == 2), "This function only makes sense for projected data."
        
        fig = plt.figure()
        ax = plt.axes(projection="3d")

        X = self.domain[:, 0].flatten().tolist()
        Y = self.domain[:, 1].flatten().tolist()
        Z, sigma = self._compute_mu_and_sigma()

        ax.plot_trisurf(X, Y, np.exp(Z.tolist()), alpha=0.8)
        if plot_sigma:
            ax.plot_trisurf(X, Y, np.exp(Z + sigma).tolist(), alpha=0.2, color="#FFDD4A")
            ax.plot_trisurf(X, Y, np.exp(Z - sigma).tolist(), alpha=0.2, color="#FFDD4A")

        if len(self.behaviors) > 0:
            behs = np.array(self.behaviors)
            ax.scatter(behs[:, 0], behs[:, 1], self.times, s=200, color="k")

        ax.set_title(f"t={self.times}")
        ax.set_xlabel(f"{self.projection[0]}")
        ax.set_ylabel(f"{self.projection[1]}")
        ax.set_zlabel(f"Predicted time")

        plt.savefig(save_path)
        plt.close(fig)
    
    def view_3D_plot(self, plot_sigma=True):
        assert len(self.projection) == 2, "This function only makes sense for projected data."
        
        fig = plt.figure()
        ax = plt.axes(projection="3d")

        X = self.domain[:, 0].flatten().tolist()
        Y = self.domain[:, 1].flatten().tolist()
        Z, sigma = self._compute_mu_and_sigma()

        ax.plot_trisurf(X, Y, np.exp(Z.tolist()), alpha=0.8)
        if plot_sigma:
            ax.plot_trisurf(X, Y, np.exp(Z + sigma).tolist(), alpha=0.2, color="#FFDD4A")
            ax.plot_trisurf(X, Y, np.exp(Z - sigma).tolist(), alpha=0.2, color="#FFDD4A")

        behs = np.array(self.behaviors)
        ax.scatter(behs[:, 0], behs[:, 1], self.times, s=200, color="k")
        plt.show()
        plt.close()

    def save_as_generation(self, save_path):
        """
        How should we deal with projections?
        solution, everything that satisfies the query
        will get casted to that performance.

        i.e. perf(b1, b2, .) = perf(b1, b2)
        """
        # The idea would be to compute the real map
        # associate it with the domain and its values
        # and save it into a JSON file.

        mu, _ = self._compute_mu_and_sigma()
        # mu is ordered according to self.domain

        # We could simply load up the prior, iterate
        # through the document and change perf. Easy as that.

        with open(self.path) as fp:
            gen = json.load(fp)
        
        for k, v in gen.items():
            if v["solution"] is not None:
                center = json.loads(k)
                projected_point = self._project_point(center)
                index = self.domain.tolist().index(projected_point)

                # Overwrite perf in v.
                v["performance"] = mu[index]
        
        # At this point, gen is updated.
        with open(save_path, "w") as fp:
            json.dump(gen, fp)
