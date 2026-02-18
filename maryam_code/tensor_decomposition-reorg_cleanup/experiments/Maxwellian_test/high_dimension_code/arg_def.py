import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()

    # Spatial domain arguments
    parser.add_argument(
        '--x_a',
        type=float,
        default=-1,
        help='Lower bound of the spatial domain x(default: -1)')
    parser.add_argument(
        '--x_b',
        type=float,
        default=1,
        help='Upper bound of the spatial domain x(default: 1)')
    parser.add_argument(
        '--y_a',
        type=float,
        default=-1,
        help='Lower bound of the spatial domain y(default: -1)')
    parser.add_argument(
        '--y_b',
        type=float,
        default=1,
        help='Upper bound of the spatial domain y(default: 1)')
    parser.add_argument(
        '--z_a',
        type=float,
        default=-1,
        help='Lower bound of the spatial domain z(default: -1)')
    parser.add_argument(
        '--z_b',
        type=float,
        default=1,
        help='Upper bound of the spatial domain z(default: 1)')
    parser.add_argument(
        '--num_internal_knots_x',
        type=int,
        default=16,
        help='Number of internal knots for spatial basis x(default: 16)')
    parser.add_argument(
        '--num_internal_knots_y',
        type=int,
        default=16,
        help='Number of internal knots for spatial basis y(default: 16)')
    parser.add_argument(
        '--num_internal_knots_z',
        type=int,
        default=16,
        help='Number of internal knots for spatial basis z(default: 16)')
    parser.add_argument(
        '--degree_x',
        type=int,
        default=3, help='Degree of spatial basis functions x(default: 3)')
    parser.add_argument(
        '--degree_y',
        type=int,
        default=3, help='Degree of spatial basis functions y(default: 3)')
    parser.add_argument(
        '--degree_z',
        type=int,
        default=3, help='Degree of spatial basis functions z(default: 3)')
    # Velocity domain arguments
    parser.add_argument(
        '--vx_a',
        type=float,
        default=-10,
        help='Lower bound of the velocity domain vx(default: -10)')
    parser.add_argument(
        '--vx_b',
        type=float,
        default=10,
        help='Upper bound of the velocity domain vx(default: 10)')
    parser.add_argument(
        '--vy_a',
        type=float,
        default=-10,
        help='Lower bound of the velocity domain vy(default: -10)')
    parser.add_argument(
        '--vy_b',
        type=float,
        default=10,
        help='Upper bound of the velocity domain vy(default: 10)')
    parser.add_argument(
        '--vz_a',
        type=float,
        default=-10,
        help='Lower bound of the velocity domain z(default: -10)')
    parser.add_argument(
        '--vz_b',
        type=float,
        default=10,
        help='Upper bound of the velocity domain z(default: 10)')
    parser.add_argument(
        '--num_internal_knots_vx',
        type=int,
        default=16,
        help='Number of internal knots for velocity basis vx(default: 16)')
    parser.add_argument(
        '--num_internal_knots_vy',
        type=int,
        default=16,
        help='Number of internal knots for velocity basis vy(default: 16)')
    parser.add_argument(
        '--num_internal_knots_vz',
        type=int,
        default=16,
        help='Number of internal knots for velocity basis vz(default: 16)')
    parser.add_argument(
        '--degree_vx',
        type=int,
        default=3,
        help='Degree of velocity basis functions vx (default: 3)')
    parser.add_argument(
        '--degree_vy',
        type=int,
        default=3,
        help='Degree of velocity basis functions vx (default: 3)')
    parser.add_argument(
        '--degree_vz',
        type=int,
        default=3,
        help='Degree of velocity basis functions vz (default: 3)')
    # Sampling arguments
    parser.add_argument(
        '--number_samples',
        type=int,
        default=1000,
        help='Number of samples for rejection sampling (default: 1000)')
    parser.add_argument(
        '--x_grid_size',
        type=int,
        default=1000,
        help='Number of points in the spatial grid (default: 1000)')
    parser.add_argument(
        '--v_grid_size',
        type=int,
        default=1000,
        help='Number of points in the velocity grid (default: 1000)')
    parser.add_argument(
        '--D',
        type=int,
        default=1,
        help='Dimentionality of spatial and velocity (default: 1)')
    parser.add_argument(
        '--basis_functions_spatial',
        default= 'Chebyshev',
        metavar='string',
        choices=['B-spline',
                 'Chebyshev',
                 'Legendre',
                 'Fourier'
                ],
        help='choose spatial basis functions to test, available: B-spline, Chebyshev, Legendre, Fourier (default: B-spline)')
    parser.add_argument(
        '--basis_functions_velocity',
        default= 'Chebyshev',
        metavar='string',
        choices=['B-spline',
                 'Chebyshev',
                 'Legendre',
                 'Fourier'
                ],
        help='choose velocity basis functions to test, available: B-spline, Chebyshev, Legendre, Fourier (default: B-spline)')
    return parser.parse_args()
