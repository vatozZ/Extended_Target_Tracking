from ett_filter import *
from Measurement_Model import generate_measurement_2D
from ett_constants import n_simulation, xlim, ylim

def main():

    xk, Pk, vk, Xk = xk_init, Pk_init, vk_init, Xk_init

    simulate_samples = [1, 2, 3, 4, 5, 6]

    IoU = [] # intersection of union  (Target Extent over Ground Truth)

    for i in range(n_simulation):

        # get mesurements
        ZK = get_Zk(n_sim=i, measurement_matrix=measurement_matrix)

        # get centroid of the measurements, and scattering matrix of the measurements
        zk, Zk, nk = calculate_mean_and_covariance(Zk=ZK)

        # PREDICTION ###
        xk, Pk, vk, Xk = predict_ETT(xk=xk, Pk=Pk, vk=vk, Xk=Xk)

        # UPDATE ####
        xk, Pk, Xk, vk = update_ETT(xk=xk, Pk=Pk, vk=vk, Xk=Xk, zk=zk, Zk=Zk, nk=nk)

        intersection_of_union = calculate_error(xk, Xk, ZK, i, ground_truth_matrix=ground_truth_matrix)

        IoU.append(intersection_of_union)

        if i in simulate_samples:
            plot_filter(xk, Xk, ZK, i, ground_truth_matrix=ground_truth_matrix, )


    plt.xlim(xlim)
    plt.ylim(ylim)

    # plt.savefig('ETT_Contour_Measurement.png', dpi=1200, bbox_inches='tight')
    plt.show()

    plt.figure()
    plt.title('Intersection of Union')
    plt.plot(IoU)
    plt.ylabel('Meter Square')
    plt.show()


if __name__ == "__main__":

    legend_labels = ['xk', 'GT', 'Xk']

    measurement_matrix, ground_truth_matrix = generate_measurement_2D()

    px, py, Vx, Vy, ax, ay = 1.0, 1.0, 1.0, 1.0, 0.1, 0.1
    s, d = 3, 2

    xk_init = np.array([px, py, Vx, Vy, ax, ay]).reshape(s *d, 1)

    std_pose, std_velocity, std_acceleration = 1.0, 1.0, 1.0

    Pk_init = np.diag([std_pose, std_velocity, std_acceleration])

    Xk_init = array([[1.0, 1e-5],
                     [1e-5, 1.0]]).reshape(d, d)

    vk_init = 2

    main()