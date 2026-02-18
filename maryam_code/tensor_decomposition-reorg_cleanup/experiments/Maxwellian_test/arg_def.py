import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()

    # Spatial domain arguments
    parser.add_argument(
        '--x_a',
        type=float,
        default=-1,
        help='Lower bound of the spatial domain (default: -1)')
    parser.add_argument(
        '--x_b',
        type=float,
        default=1,
        help='Upper bound of the spatial domain (default: 1)')
    parser.add_argument(
        '--num_internal_knots_x',
        type=int,
        default=16,
        help='Number of internal knots for spatial basis (default: 16)')
    parser.add_argument(
        '--degree_x',
        type=int,
        default=3, help='Degree of B-splines for spatial basis (default: 3)')

    # Velocity domain arguments
    parser.add_argument(
        '--v_a',
        type=float,
        default=-10,
        help='Lower bound of the velocity domain (default: -10)')
    parser.add_argument(
        '--v_b',
        type=float,
        default=10,
        help='Upper bound of the velocity domain (default: 10)')
    parser.add_argument(
        '--num_internal_knots_v',
        type=int,
        default=16,
        help='Number of internal knots for velocity basis (default: 16)')
    parser.add_argument(
        '--degree_v',
        type=int,
        default=3,
        help='Degree of B-splines for velocity basis (default: 3)')

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
    return parser.parse_args()
