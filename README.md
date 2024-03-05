
##class FEKFSLAMFeature(MapFeature): 
---
This class extends the `MapFeature` class to implement the Feature EKF SLAM algorithm.
The `MapFeature` class is a base class providing support to localize the robot using a map of point features.
The main difference between FEKMBL and FEAKFSLAM is that the former uses the robot pose as a state variable,
while the latter uses the robot pose and the feature map as state variables. This means that few methods provided by
class need to be overridden to gather the information from state vector instead that from the deterministic map.

***
#### def hfj(self, xk_bar, Fj):
...

This method implements the direct observation model for a single feature observation $z_{f_i}$ , so it implements its related
observation function (see eq. `eq-FEKFSLAM-hfj`). For a single feature observation $z_{f_i}$ of the feature $^Nx_{F_j}$ the method computes its
expected observation from the current robot pose $^Nx_B$.
This function uses a generic implementation through the following equation:

$$
z_{f_i}=h_{Fj}(x_k,v_k)=s2o(\ominus ^Nx_B \boxplus ^Nx_{F_j}) + v_{fi_k}
$$

Where $^Nx_B$ is the robot pose and $^Nx_{F_j}$ are both included within the state vector:

$$
x_k=[^Nx_B^T~\cdots~^Nx_{F_j}~\cdots~^Nx_{F_{nf}}]^T
$$

and `s2o` is a conversion function from the store representation to the observation representation.

The method is called by `FEKFSLAM.hf` to compute the expected observation for each feature
observation contained in the observation vector $z_f=[z_{f_1}^T~\cdots~z_{f_i}^T~\cdots~z_{f_{n_zf}}^T]^T$.

- **Parameters**:
  - `xk_bar`: mean of the predicted state vector
  - `Fj`: map index of the observed feature.
- **Returns**:
  - expected observation of the feature $^Nx_{F_j}$
---
#### def Jhfjx():
####      ....
Jacobian of the single feature direct observation model `hfj` (eq. `eq-FEKFSLAM-hfj`) with respect to the state vector $\bar{x}_k$:

$$
x_k=[^Nx_B^T~\cdots~^Nx_{F_j}~\cdots~^Nx_{F_{nf}}]^T
$$

$$
J_{hfjx}=\frac{\partial h_{f_{zfi}}({x}_k, v_k)}{\partial {x}_k}=
\frac{\partial s2o(\ominus ^Nx_B \boxplus ^Nx_{F_j})+v_{fi_k}}{\partial {x}_k}
$$

$$
👉
\begin{bmatrix}
\frac{\partial{h_{F_j}(x_k,v_k)}}{ \partial {{}^Nx_{B_k}}} & \frac{\partial{h_{F_j}(x_k,v_k)}}{ \partial {{}^Nx_{F_1}}} & \cdots &\frac{\partial{h_{F_j}(x_k,v_k)}}{ \partial {{}^Nx_{F_j}}} & \cdots & \frac{\partial{h_{F_j}(x_k,v_k)}}{ \partial {{}^Nx_{F_n}} }
\end{bmatrix}
$$

$$
👉
\begin{bmatrix}
J_{s2o}{J_{1\boxplus} J_\ominus} & {0} & \cdots & J_{s2o}{J_{2\boxplus}} & \cdots &{0}
\end{bmatrix}
$$

where we have used the abbreviation:

$$
J_{s2o} \equiv J_{s2o}(\ominus ^Nx_B \boxplus^Nx_{F_j})
$$

$$
J_{1\boxplus} \equiv J_{1\boxplus}(\ominus ^Nx_B,^Nx_{F_j} )
$$

$$
J_{2\boxplus} \equiv J_{2\boxplus}(\ominus ^Nx_B,^Nx_{F_j} )
$$

- **Parameters**:
  - `xk`: state vector mean
  - `Fj`: map index of the observed feature
- **Returns**:
  - Jacobian matrix defined in eq. `eq-Jhfjx`