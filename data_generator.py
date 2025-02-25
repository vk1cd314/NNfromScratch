import numpy as np

class DataGenerator:
    @staticmethod
    def circles(n_samples=1000, noise=0.1, factor=0.5):
        """
        Generate circles dataset with non-linear decision boundary
        
        Parameters:
        - n_samples: number of points to generate
        - noise: standard deviation of Gaussian noise
        - factor: scale factor for radius of inner circle
        """
        n_samples_out = n_samples // 2
        n_samples_in = n_samples - n_samples_out
        
        # Generate outer circle
        linspace = np.linspace(0, 2 * np.pi, n_samples_out, endpoint=False)
        outer_x = np.cos(linspace)
        outer_y = np.sin(linspace)
        outer_points = np.c_[outer_x, outer_y]
        
        # Generate inner circle
        linspace = np.linspace(0, 2 * np.pi, n_samples_in, endpoint=False)
        inner_x = factor * np.cos(linspace)
        inner_y = factor * np.sin(linspace)
        inner_points = np.c_[inner_x, inner_y]
        
        # Add noise
        outer_points += np.random.randn(n_samples_out, 2) * noise
        inner_points += np.random.randn(n_samples_in, 2) * noise
        
        # Create dataset
        X = np.vstack([outer_points, inner_points])
        y = np.hstack([np.zeros(n_samples_out), np.ones(n_samples_in)])
        
        # Shuffle the dataset
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        return X[indices], y[indices].reshape(-1, 1)
    
    @staticmethod
    def moons(n_samples=1000, noise=0.1):
        """
        Generate two interleaving half circles (moons dataset)
        
        Parameters:
        - n_samples: number of points to generate
        - noise: standard deviation of Gaussian noise
        """
        n_samples_out = n_samples // 2
        n_samples_in = n_samples - n_samples_out
        
        # Generate outer moon
        outer_circ_x = np.cos(np.linspace(0, np.pi, n_samples_out))
        outer_circ_y = np.sin(np.linspace(0, np.pi, n_samples_out))
        outer_points = np.vstack([outer_circ_x, outer_circ_y]).T
        
        # Generate inner moon
        inner_circ_x = 1 - np.cos(np.linspace(0, np.pi, n_samples_in))
        inner_circ_y = 1 - np.sin(np.linspace(0, np.pi, n_samples_in)) - 0.5
        inner_points = np.vstack([inner_circ_x, inner_circ_y]).T
        
        # Add noise
        outer_points += np.random.randn(n_samples_out, 2) * noise
        inner_points += np.random.randn(n_samples_in, 2) * noise
        
        # Create dataset
        X = np.vstack([outer_points, inner_points])
        y = np.hstack([np.zeros(n_samples_out), np.ones(n_samples_in)])
        
        # Shuffle the dataset
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        return X[indices], y[indices].reshape(-1, 1)
    
    @staticmethod
    def spiral(n_samples=1000, noise=0.2):
        """
        Generate spiral dataset with non-linear decision boundary
        
        Parameters:
        - n_samples: number of points to generate
        - noise: standard deviation of Gaussian noise
        """
        n = n_samples // 2
        
        def gen_spiral(delta_t, delta_r):
            t = np.sqrt(np.random.rand(n)) * 3 * np.pi
            r = 2 * t + delta_r
            x = r * np.sin(t + delta_t)
            y = r * np.cos(t + delta_t)
            return np.vstack([x, y]).T
        
        spiral1 = gen_spiral(0, 0) + np.random.randn(n, 2) * noise
        spiral2 = gen_spiral(np.pi, 0) + np.random.randn(n, 2) * noise
        
        X = np.vstack([spiral1, spiral2])
        y = np.hstack([np.zeros(n), np.ones(n)])
        
        # Shuffle the dataset
        indices = np.arange(n*2)
        np.random.shuffle(indices)
        
        return X[indices], y[indices].reshape(-1, 1)
    
    @staticmethod
    def xor(n_samples=1000, noise=0.1):
        """
        Generate XOR problem dataset
        
        Parameters:
        - n_samples: number of points to generate
        - noise: standard deviation of Gaussian noise
        """
        n_per_cluster = n_samples // 4
        
        # Generate 4 clusters for XOR pattern
        x1 = np.random.randn(n_per_cluster, 2) * 0.3 + np.array([1, 1])
        x2 = np.random.randn(n_per_cluster, 2) * 0.3 + np.array([-1, -1])
        x3 = np.random.randn(n_per_cluster, 2) * 0.3 + np.array([1, -1])
        x4 = np.random.randn(n_per_cluster, 2) * 0.3 + np.array([-1, 1])
        
        # Combine clusters
        X = np.vstack([x1, x2, x3, x4])
        
        # Assign labels (class 0 for clusters 1 and 2, class 1 for clusters 3 and 4)
        y = np.hstack([np.zeros(2*n_per_cluster), np.ones(2*n_per_cluster)])
        
        # Shuffle the dataset
        indices = np.arange(n_per_cluster*4)
        np.random.shuffle(indices)
        
        return X[indices], y[indices].reshape(-1, 1)
