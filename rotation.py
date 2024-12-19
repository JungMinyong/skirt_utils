import numpy as np

def angles_from_direction(k_obs):
    """
    Given k_obs = [kx, ky, kz], where k_obs = (-sinθ cosφ, -sinθ sinφ, -cosθ),
    solve for θ and φ.
    """
    kx, ky, kz = k_obs
    # θ = arccos(-kz)
    theta = np.arccos(-kz)
    # φ = atan2(ky, kx)
    phi = np.arctan2(ky, kx)
    return theta, phi

def y_view_0(theta, phi):
    """
    Compute y_view for ω=0:
    y_view_0 = (-cosφ cosθ, -sinφ cosθ, sinθ).
    """
    return np.array([-np.cos(phi)*np.cos(theta),
                     -np.sin(phi)*np.cos(theta),
                      np.sin(theta)])

def k_obs_from_angles(theta, phi):
    """
    k_obs = (-sinθ cosφ, -sinθ sinφ, -cosθ)
    """
    return np.array([-np.sin(theta)*np.cos(phi),
                     -np.sin(theta)*np.sin(phi),
                     -np.cos(theta)])

def compute_omega(y_view_orig, k_obs_orig, theta, phi):
    """
    Given y_view_orig and k_obs_orig and known θ, φ, find ω.
    First compute y_view_0 and x_view_0.
    ω = atan2(y_view_orig·x_view_0, y_view_orig·y_view_0)
    """
    y0 = y_view_0(theta, phi)
    # x_view_0 = k_obs_orig × y0 (ensure right-handed system)
    x0 = np.cross(k_obs_orig, y0)

    # Normalize
    x0 /= np.linalg.norm(x0)
    y0 /= np.linalg.norm(y0)

    # Compute sinω and cosω from dot products
    cos_omega = np.dot(y_view_orig, y0)
    sin_omega = np.dot(y_view_orig, x0)
    omega = np.arctan2(sin_omega, cos_omega)
    return omega

# Example usage:

# Suppose we have rotation_matrix and L_hat from your setup
# For demonstration, let's assume some rotation_matrix and L_hat:
L = np.array([1,1,0])
L_hat = L / np.linalg.norm(L)
rotation_axis = np.cross(L_hat, [0, 0, 1])
rotation_angle = np.arccos(np.dot(L_hat, [0, 0, 1]))
# Rodrigues' rotation formula to construct the rotation matrix
K = np.array([[0, -rotation_axis[2], rotation_axis[1]],
              [rotation_axis[2], 0, -rotation_axis[0]],
              [-rotation_axis[1], rotation_axis[0], 0]])
rotation_matrix = np.eye(3) + np.sin(rotation_angle) * K + (1 - np.cos(rotation_angle)) * np.dot(K, K)

Q = np.array([[0., 0., 1.],
              [1., 0., 0.],
              [0., 1., 0.]])

# Final rotation: from original to final
#rotation_matrix = (rotation_matrix.T @ Q).T


# 1) For the XY-plane view in rotated frame: (θ,φ,ω) = (0°, 0°, 90°)
theta_rot = np.radians(0.)
phi_rot = np.radians(0.)
omega_rot = np.radians(90.)

# k_obs_rot and y_view_rot in rotated frame
k_obs_rot = np.array([0,0,-1])
# y_view_rot = (0,-1,0) from the given angles
y_view_rot = np.array([0,-1,0])

# Transform to original frame
k_obs_orig = rotation_matrix @ k_obs_rot
y_view_orig = rotation_matrix @ y_view_rot

# Now find (θ', φ', ω') in original frame
theta_prime, phi_prime = angles_from_direction(k_obs_orig)
omega_prime = compute_omega(y_view_orig, k_obs_orig, theta_prime, phi_prime)

print("Face-on view in original frame:")
print("theta' [deg]:", np.degrees(theta_prime))
print("phi'   [deg]:", np.degrees(phi_prime))
print("omega' [deg]:", np.degrees(omega_prime))

# 2) For the XZ-plane view in rotated frame: (θ, φ, ω) = (90°, -90°, 0°)
theta_rot = np.radians(90.)
phi_rot = np.radians(-90.)
omega_rot = np.radians(0.)
#omega_rot = np.radians(180.)

# Compute k_obs_rot for these angles:
k_obs_rot = [0, -1, 0]#k_obs_from_angles(theta_rot, phi_rot)

# For ω=0, y_view_0_rot = (-cosφcosθ, -sinφcosθ, sinθ)
# plug in θ=90°, φ=-90°: 
# cos(90°)=0, sin(90°)=1, cos(-90°)=0, sin(-90°)=-1
# y_view_0_rot = (0,0,1)
y_view_0_rot = np.array([0,0,1]) 
y_view_rot = y_view_0_rot  # since ω=0

k_obs_orig = rotation_matrix @ k_obs_rot
y_view_orig = rotation_matrix @ y_view_rot

theta_prime, phi_prime = angles_from_direction(k_obs_orig)
omega_prime = compute_omega(y_view_orig, k_obs_orig, theta_prime, phi_prime)

print("Edge-on view in original frame:")
print("theta' [deg]:", np.degrees(theta_prime))
print("phi'   [deg]:", np.degrees(phi_prime))
print("omega' [deg]:", np.degrees(omega_prime) + 180)
