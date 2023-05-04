import os
import math
import numpy as np
import matplotlib.pyplot as plt

FOLDER_PATH = "./dataset/"

def freedman_diaconis_bins(data):
    # Calculate the IQR of the data
    iqr = np.percentile(data, 75) - np.percentile(data, 25)
    
    # Calculate the bin width
    bin_width = 2 * iqr / (len(data)**(1/3))
    
    # Calculate the number of bins
    bins = np.ceil((np.max(data) - np.min(data)) / bin_width)
    
    return int(bins)

class Sample:
    
    def __init__(self, center=None):
        self.center: Point3D = center
        
        # Initialize other points as empty
        self.head: Point3D = None
        self.left_hand: Point3D = None
        self.right_hand: Point3D = None
        self.left_foot: Point3D = None
        self.right_foot: Point3D = None
        
        self.left_elbow: Point3D = None
        self.right_elbow: Point3D = None
        self.left_knee: Point3D = None
        self.right_knee: Point3D = None
        
        self.distances_to_center: list[float] = []
        self.angles_between_extremities: list[float] = []
        
        self.frame_id: int = None
    
    def set_center(self, point):
        self.center = point
    
    def set_head(self, point):
        self.head = point
        
    def set_left_hand(self, point):
        self.left_hand = point
        
    def set_right_hand(self, point):
        self.right_hand = point
        
    def set_left_foot(self, point):
        self.left_foot = point
        
    def set_right_foot(self, point):
        self.right_foot = point
    
    def __repr__(self):
        output = f"\n\tframe: {self.frame_id}\n\tcenter: {self.center}\n\tpoints: "
        output += str([self.head, self.left_hand, self.right_hand, self.left_foot, self.right_foot])
        output += f"\n\tdistances: {self.distances_to_center}"
        output += f"\n\tangles: {self.angles_between_extremities}"
        return f"\nSample({output}\n)"

class Point3D:
    
    def __init__(self, x=None, y=None, z=None):
        self.x = x
        self.y = y
        self.z = z
    
    def __repr__(self):
        return f"({self.x}, {self.y}, {self.z})"
    
    def __str__(self):
        return f"({self.x}, {self.y}, {self.z})"

    def set_point(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
    
    def distance_to_other_point(self, other):
        if isinstance(other, Point3D):
            return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)
        else:
            raise TypeError("Input must be an instance of Point3D class")

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

def load_from_file(filename, debug=False):
    '''
    Loads one file into a dictionary. 
    '''
    data = {
        "observations": []
    }
    
    # Access and store metadata from filename
    metadata = filename.split(os.path.sep)[-1].split("_")[:3] # isolate parts we want and discard all else
    activity, subject, trial = [int(x[1:]) for x in metadata] # unpack and remove _'s
    
    data["activity"] = activity
    data["subject"] = subject
    data["trial"] = trial
    
    # Read in observations from file
    with open(filename, 'r') as file:
        line = file.readline()
        while line:
            line = file.readline().strip().split(" ")
            if len(line) != 5: break
            
            frame_id, joint_id, pos_x, pos_y, pos_z = line
            frame_id = int(frame_id)
            joint_id = int(joint_id)
            pos_x, pos_y, pos_z = float(pos_x), float(pos_y), float(pos_z)
            
            observation = {
                "frame_id": frame_id,
                "joint_id": joint_id,
                "position": (pos_x, pos_y, pos_z)    
            }
            data["observations"].append(observation)
    
    return data, activity

def angle_between_points(p1: Point3D, p2: Point3D):
    dot_product = p1.dot(p2)
    p1_squared_sum = p1.x**2 + p1.y**2 + p1.z**2
    p2_squared_sum = p2.x**2 + p2.y**2 + p2.z**2
    return math.acos(dot_product / math.sqrt( p1_squared_sum * p2_squared_sum ))

def RAD_representation(dataset: list[dict], activities, testing=False, ONE_ITER=False):
    '''
    dataset is an entire list of files worth of data
    '''
    
    '''
    1: center
    4: head
    8: left hand
    12: right hand
    16: left foot
    20: right foot
    '''
    
    filepath = ""
    if testing: filepath = "./Outputs/Testing/rad_d1_test.txt"
    else: filepath = "./Outputs/Training/rad_d1_train.txt"
    with open(filepath, "w") as file:
        file.write("")
    
    star_joint_ids = [4, 16, 20, 8, 12]
    center_id = 1
    
    # filter out anything that isn't a relevant joint
    filtered_dataset = [
        {
            "observations": [
                observation
                for observation in data["observations"]
                if observation["joint_id"] in star_joint_ids or observation["joint_id"] == center_id
            ]
        }
        for data in dataset
    ]
    
    samples = []
    
    for i, instance in enumerate(filtered_dataset):
        # instance is a dict
        
        samples = []
        current_sample = []
        observations = instance["observations"]
        activity = activities[i]
        
        for observation in observations: # 5 iterations
            # observation is a dict
            
            frame_id: int = observation["frame_id"]
            joint_id: int = observation["joint_id"]
            position: tuple = observation["position"]
            
            if joint_id == center_id: 
                current_sample = [observation]
            else:
                current_sample.append(observation)
            
            # If on last joint, then sample is complete
            if joint_id == 20:
                
                if len(current_sample) != 6: continue
                
                center = Point3D(*current_sample[0]["position"])
                sample = Sample(center)
                
                # add other points to the sample
                head = Point3D(*current_sample[1]["position"])
                left_hand = Point3D(*current_sample[2]["position"])
                right_hand = Point3D(*current_sample[3]["position"])
                left_foot = Point3D(*current_sample[4]["position"])
                right_foot = Point3D(*current_sample[5]["position"])
                
                sample.set_head(head)
                sample.set_left_hand(left_hand)
                sample.set_right_hand(right_hand)
                sample.set_left_foot(left_foot)
                sample.set_right_foot(right_foot)
                
                sample.frame_id = frame_id
                
                # Compute and store distances between body extremities to body center
                distances = [center.distance_to_other_point(point) for point in [head, left_hand, right_hand, left_foot, right_foot]]
                sample.distances_to_center = distances

                # Compute and store angles between two adjacent body extremities
                angles = [angle_between_points(head, left_hand),
                            angle_between_points(left_hand, left_foot),
                            angle_between_points(left_foot, right_foot),
                            angle_between_points(right_foot, right_hand),
                            angle_between_points(right_hand, head)]
                sample.angles_between_extremities = angles
                
                samples.append(sample)
        
        # Histograms
        
        T = len(samples)
        
        distances_to_head = []
        distances_to_left_hand = []
        distances_to_right_hand = []
        distances_to_left_foot = []
        distances_to_right_foot = []
        
        angles_head_to_left_hand = []
        angles_left_hand_to_left_foot = []
        angles_left_foot_to_right_foot = []
        angles_right_foot_to_right_hand = []
        angles_right_hand_to_head = []
        
        for sample in samples:
            distances_to_head.append(sample.distances_to_center[0])
            distances_to_left_hand.append(sample.distances_to_center[1])
            distances_to_right_hand.append(sample.distances_to_center[2])
            distances_to_left_foot.append(sample.distances_to_center[3])
            distances_to_right_foot.append(sample.distances_to_center[4])
            
            angles_head_to_left_hand.append(sample.angles_between_extremities[0])
            angles_left_hand_to_left_foot.append(sample.angles_between_extremities[1])
            angles_left_foot_to_right_foot.append(sample.angles_between_extremities[2])
            angles_right_foot_to_right_hand.append(sample.angles_between_extremities[3])
            angles_right_hand_to_head.append(sample.angles_between_extremities[4])
            
        distances_list = [
            distances_to_head, 
            distances_to_left_hand, 
            distances_to_right_hand, 
            distances_to_left_foot, 
            distances_to_right_foot
        ]

        angles_list = [
            angles_head_to_left_hand, 
            angles_left_hand_to_left_foot, 
            angles_left_foot_to_right_foot, 
            angles_right_foot_to_right_hand, 
            angles_right_hand_to_head
        ]
        
        # bins_distances = [freedman_diaconis_bins(distances) for distances in distances_list]
        # bins_angles = [freedman_diaconis_bins(angles) for angles in angles_list]
        
        N = 9  # Number of bins for distances
        M = 10  # Number of bins for angles

        # Compute histograms
        # hist_distances_head, _ = np.histogram(distances_to_head, bins=bins_distances[0])
        # hist_distances_left_hand, _ = np.histogram(distances_to_left_hand, bins=bins_distances[1])
        # hist_distances_right_hand, _ = np.histogram(distances_to_right_hand, bins=bins_distances[2])
        # hist_distances_left_foot, _ = np.histogram(distances_to_left_foot, bins=bins_distances[3])
        # hist_distances_right_foot, _ = np.histogram(distances_to_right_foot, bins=bins_distances[4])
        hist_distances_head, _ = np.histogram(distances_to_head, bins=N)
        hist_distances_left_hand, _ = np.histogram(distances_to_left_hand, bins=N)
        hist_distances_right_hand, _ = np.histogram(distances_to_right_hand, bins=N)
        hist_distances_left_foot, _ = np.histogram(distances_to_left_foot, bins=N)
        hist_distances_right_foot, _ = np.histogram(distances_to_right_foot, bins=N)

        hist_angles_head_left_hand, _ = np.histogram(angles_head_to_left_hand, bins=M)
        hist_angles_left_hand_left_foot, _ = np.histogram(angles_left_hand_to_left_foot, bins=M)
        hist_angles_left_foot_right_foot, _ = np.histogram(angles_left_foot_to_right_foot, bins=M)
        hist_angles_right_foot_right_hand, _ = np.histogram(angles_right_foot_to_right_hand, bins=M)
        hist_angles_right_hand_head, _ = np.histogram(angles_right_hand_to_head, bins=M)
        
        # Normalize histograms
        normalized_hist_distances_head = hist_distances_head / T
        normalized_hist_distances_left_hand = hist_distances_left_hand / T
        normalized_hist_distances_right_hand = hist_distances_right_hand / T
        normalized_hist_distances_left_foot = hist_distances_left_foot / T
        normalized_hist_distances_right_foot = hist_distances_right_foot / T

        normalized_hist_angles_head_left_hand = hist_angles_head_left_hand / T
        normalized_hist_angles_left_hand_left_foot = hist_angles_left_hand_left_foot / T
        normalized_hist_angles_left_foot_right_foot = hist_angles_left_foot_right_foot / T
        normalized_hist_angles_right_foot_right_hand = hist_angles_right_foot_right_hand / T
        normalized_hist_angles_right_hand_head = hist_angles_right_hand_head / T
        
        # Concatenate all normalized histograms
        concatenated_histograms = np.concatenate((
            normalized_hist_distances_head,
            normalized_hist_distances_left_hand,
            normalized_hist_distances_right_hand,
            normalized_hist_distances_left_foot,
            normalized_hist_distances_right_foot,
            normalized_hist_angles_head_left_hand,
            normalized_hist_angles_left_hand_left_foot,
            normalized_hist_angles_left_foot_right_foot,
            normalized_hist_angles_right_foot_right_hand,
            normalized_hist_angles_right_hand_head
        ))
        
        
        # Create weights array for normalization
        weights = np.ones_like(distances_to_head) / T

        if ONE_ITER:

            # Colored and overlayed plots: 
            # # Plot normalized histograms for distances
            # plt.figure()
            # for distances, label, bins in zip(distances_list, ["Head", "Left Hand", "Right Hand", "Left Foot", "Right Foot"], bins_distances):
            #     plt.hist(distances, bins=bins, weights=weights, alpha=0.5, label=label)
            # plt.xlabel("Distance")
            # plt.ylabel("Normalized Frequency")
            # plt.legend()
            # plt.title("Normalized Histograms of Distances")

            # # Plot normalized histograms for angles
            # plt.figure()
            # for angles, label, bins in zip(angles_list, ["Head-Left Hand", "Left Hand-Left Foot", "Left Foot-Right Foot", "Right Foot-Right Hand", "Right Hand-Head"], bins_angles):
            #     plt.hist(angles, bins=bins, weights=weights, alpha=0.5, label=label)
            # plt.xlabel("Angle")
            # plt.ylabel("Normalized Frequency")
            # plt.legend()
            # plt.title("Normalized Histograms of Angles")

            # # Display the plots
            # plt.show()
            
            return 
            
            # Plot normalized histograms for distances
            fig, axes = plt.subplots(1, 5, figsize=(15, 3))
            labels = ["Head", "Left Hand", "Right Hand", "Left Foot", "Right Foot"]

            for i, (distances, label, bins) in enumerate(zip(distances_list, labels, bins_distances)):
                axes[i].hist(distances, bins=bins, weights=weights, alpha=0.5, label=label)
                axes[i].set_xlabel("Distance")
                axes[i].set_ylabel("Normalized Frequency")
                axes[i].set_title(f"Normalized Histogram of {label} Distances", fontsize=8)

            plt.tight_layout()
            plt.show()

            # Plot normalized histograms for angles
            fig, axes = plt.subplots(1, 5, figsize=(15, 3))
            labels = ["Head-Left Hand", "Left Hand-Left Foot", "Left Foot-Right Foot", "Right Foot-Right Hand", "Right Hand-Head"]

            for i, (angles, label, bins) in enumerate(zip(angles_list, labels, bins_angles)):
                axes[i].hist(angles, bins=bins, weights=weights, alpha=0.5, label=label)
                axes[i].set_xlabel("Angle")
                axes[i].set_ylabel("Normalized Frequency")
                axes[i].set_title(f"Normalized Histogram of {label} Angles", fontsize=8)

            plt.tight_layout()
            plt.show()

            return 
        
        concatenated_histograms_str = " ".join([str(val) for val in concatenated_histograms])
        with open(filepath, "a") as f:
            # Append concatenated_histograms to file
            f.write(concatenated_histograms_str + " " + str(activity) + "\n")

def custom_representation(dataset: list[dict], activities, testing=False, ONE_ITER=False):
    '''
    dataset is an entire list of files worth of data
    '''
    
    '''
    1: center
    4: head
    8: left hand
    12: right hand
    16: left foot
    20: right foot
    '''
    
    filepath = ""
    if testing: filepath = "./Outputs/Testing/cust_d1_test.txt"
    else: filepath = "./Outputs/Training/cust_d1_train.txt"
    with open(filepath, "w") as file:
        file.write("")
    
    star_joint_ids = [4, 16, 20, 8, 12, 6, 10, 14, 18]
    center_id = 1
    
    # filter out anything that isn't a relevant joint
    filtered_dataset = [
        {
            "observations": [
                observation
                for observation in data["observations"]
                if observation["joint_id"] in star_joint_ids or observation["joint_id"] == center_id
            ]
        }
        for data in dataset
    ]
    
    samples = []
    
    # Reduce each obervation to just the data we care about
    for i, instance in enumerate(filtered_dataset):
        # instance is a dict
        
        print("Handling instance")
        
        samples = []
        current_sample = []
        observations = instance["observations"]
        activity = activities[i]
        
        for observation in observations: # 9 iterations
            # observation is a dict
            
            frame_id: int = observation["frame_id"]
            joint_id: int = observation["joint_id"]
            position: tuple = observation["position"]
            
            if joint_id == center_id: 
                current_sample = [observation]
            else:
                current_sample.append(observation)
            
            # If on last joint, then sample is complete
            if joint_id == 20:
                
                if len(current_sample) != 10: continue
                
                center = Point3D(*current_sample[0]["position"])
                sample = Sample(center)
                
                # add other points to the sample
                head = Point3D(*current_sample[1]["position"])
                left_elbow = Point3D(*current_sample[2]["position"])
                left_hand = Point3D(*current_sample[3]["position"])
                right_elbow = Point3D(*current_sample[4]["position"])
                right_hand = Point3D(*current_sample[5]["position"])
                left_knee = Point3D(*current_sample[6]["position"])
                left_foot = Point3D(*current_sample[7]["position"])
                right_knee = Point3D(*current_sample[8]["position"])
                right_foot = Point3D(*current_sample[9]["position"])
                
                sample.set_head(head)
                sample.set_left_hand(left_hand)
                sample.set_right_hand(right_hand)
                sample.set_left_foot(left_foot)
                sample.set_right_foot(right_foot)
                sample.left_elbow = left_elbow
                sample.right_elbow = right_elbow
                sample.left_knee = left_knee
                sample.right_knee = right_knee
                
                sample.frame_id = frame_id
                
                # Compute and store distances between body extremities to body center
                distances = [center.distance_to_other_point(point) for point in [head, left_elbow, left_hand, left_knee, right_elbow, right_hand, right_knee, left_foot, right_foot]]
                sample.distances_to_center = distances

                # Compute and store angles between two adjacent body extremities
                angles = [angle_between_points(head, left_elbow),
                            angle_between_points(left_elbow, left_hand),
                            angle_between_points(left_hand, left_knee),
                            angle_between_points(left_knee, left_foot),
                            angle_between_points(left_foot, right_foot),
                            angle_between_points(right_foot, right_knee),
                            angle_between_points(right_knee, right_hand),
                            angle_between_points(right_hand, right_elbow),
                            angle_between_points(right_elbow, head)
                            ]
                sample.angles_between_extremities = angles
                
                samples.append(sample)
        
        # Histograms
        
        T = len(samples)
        
        distances_to_head = []
        distances_to_left_elbow = []
        distances_to_left_hand = []
        distances_to_right_elbow = []
        distances_to_right_hand = []
        distances_to_left_knee = []
        distances_to_left_foot = []
        distances_to_right_knee = []
        distances_to_right_foot = []
        
        
        angles_head_to_left_elbow = []
        angles_left_elbow_to_left_hand = []
        angles_left_hand_to_left_knee = []
        angles_left_knee_to_left_foot = []
        angles_left_foot_to_right_foot = []
        angles_right_foot_to_right_knee = []
        angles_right_knee_to_right_hand = []
        angles_right_hand_to_right_elbow = []
        angles_right_elbow_to_head = []
        
        
        for sample in samples:
            distances_to_head.append(sample.distances_to_center[0])
            distances_to_left_elbow.append(sample.distances_to_center[1])
            distances_to_left_hand.append(sample.distances_to_center[2])
            distances_to_right_elbow.append(sample.distances_to_center[3])
            distances_to_right_hand.append(sample.distances_to_center[4])
            distances_to_left_knee.append(sample.distances_to_center[5])
            distances_to_left_foot.append(sample.distances_to_center[6])
            distances_to_right_knee.append(sample.distances_to_center[7])
            distances_to_right_foot.append(sample.distances_to_center[8])
            
            
            angles_head_to_left_elbow.append(sample.angles_between_extremities[0])
            angles_left_elbow_to_left_hand.append(sample.angles_between_extremities[1])
            angles_left_hand_to_left_knee.append(sample.angles_between_extremities[2])
            angles_left_knee_to_left_foot.append(sample.angles_between_extremities[3])
            angles_left_foot_to_right_foot.append(sample.angles_between_extremities[4])
            angles_right_foot_to_right_knee.append(sample.angles_between_extremities[5])
            angles_right_knee_to_right_hand.append(sample.angles_between_extremities[6])
            angles_right_hand_to_right_elbow.append(sample.angles_between_extremities[7])
            angles_right_elbow_to_head.append(sample.angles_between_extremities[8])
        
        N = 8  # Number of bins for distances
        M = 11  # Number of bins for angles

        # Compute histograms
        hist_distances_head, _ = np.histogram(distances_to_head, bins=N)
        hist_distances_left_elbow, _ = np.histogram(distances_to_left_elbow, bins=N)
        hist_distances_left_hand, _ = np.histogram(distances_to_left_hand, bins=N)
        hist_distances_right_elbow, _ = np.histogram(distances_to_right_elbow, bins=N)
        hist_distances_right_hand, _ = np.histogram(distances_to_right_hand, bins=N)
        hist_distances_left_knee, _ = np.histogram(distances_to_left_knee, bins=N)
        hist_distances_left_foot, _ = np.histogram(distances_to_left_foot, bins=N)
        hist_distances_right_knee, _ = np.histogram(distances_to_right_knee, bins=N)
        hist_distances_right_foot, _ = np.histogram(distances_to_right_foot, bins=N)
        
        hist_angles_head_left_elbow, _ = np.histogram(angles_head_to_left_elbow, bins=M)
        hist_angles_left_elbow_left_hand, _ = np.histogram(angles_left_elbow_to_left_hand, bins=M)
        hist_angles_left_hand_left_knee, _ = np.histogram(angles_left_hand_to_left_knee, bins=M)
        hist_angles_left_knee_left_foot, _ = np.histogram(angles_left_knee_to_left_foot, bins=M)
        hist_angles_left_foot_right_foot, _ = np.histogram(angles_left_foot_to_right_foot, bins=M)
        hist_angles_right_foot_right_knee, _ = np.histogram(angles_right_foot_to_right_knee, bins=M)
        hist_angles_right_knee_right_hand, _ = np.histogram(angles_right_knee_to_right_hand, bins=M)
        hist_angles_right_hand_right_elbow, _ = np.histogram(angles_right_hand_to_right_elbow, bins=M)
        hist_angles_right_elbow_head, _ = np.histogram(angles_right_elbow_to_head, bins=M)
        
        # Normalize histograms
        normalized_hist_distances_head = hist_distances_head / T
        normalized_hist_distances_left_elbow = hist_distances_left_elbow / T
        normalized_hist_distances_left_hand = hist_distances_left_hand / T
        normalized_hist_distances_right_elbow = hist_distances_right_elbow / T
        normalized_hist_distances_right_hand = hist_distances_right_hand / T
        normalized_hist_distances_left_knee = hist_distances_left_knee / T
        normalized_hist_distances_left_foot = hist_distances_left_foot / T
        normalized_hist_distances_right_knee = hist_distances_right_knee / T
        normalized_hist_distances_right_foot = hist_distances_right_foot / T
        
        normalized_hist_angles_head_left_elbow = hist_angles_head_left_elbow / T
        normalized_hist_angles_left_elbow_left_hand = hist_angles_left_elbow_left_hand / T
        normalized_hist_angles_left_hand_left_knee = hist_angles_left_hand_left_knee / T
        normalized_hist_angles_left_knee_left_foot = hist_angles_left_knee_left_foot / T
        normalized_hist_angles_left_foot_right_foot = hist_angles_left_foot_right_foot / T
        normalized_hist_angles_right_foot_right_knee = hist_angles_right_foot_right_knee / T
        normalized_hist_angles_right_knee_right_hand = hist_angles_right_knee_right_hand / T
        normalized_hist_angles_right_hand_right_elbow = hist_angles_right_hand_right_elbow / T
        normalized_hist_angles_right_elbow_head = hist_angles_right_elbow_head / T
        
        
        # Concatenate all normalized histograms
        concatenated_histograms = np.concatenate((
            normalized_hist_distances_head,
            normalized_hist_distances_left_elbow,
            normalized_hist_distances_left_hand,
            normalized_hist_distances_right_elbow,
            normalized_hist_distances_right_hand,
            normalized_hist_distances_left_knee,
            normalized_hist_distances_left_foot,
            normalized_hist_distances_right_knee,
            normalized_hist_distances_right_foot,
            normalized_hist_angles_head_left_elbow,
            normalized_hist_angles_left_elbow_left_hand,
            normalized_hist_angles_left_hand_left_knee,
            normalized_hist_angles_left_knee_left_foot,
            normalized_hist_angles_left_foot_right_foot,
            normalized_hist_angles_right_foot_right_knee,
            normalized_hist_angles_right_knee_right_hand,
            normalized_hist_angles_right_hand_right_elbow,
            normalized_hist_angles_right_elbow_head
        ))
        
        # Create weights array for normalization
        weights = np.ones_like(distances_to_head) / T
        
        distances_list = [
            distances_to_head, 
            distances_to_left_elbow, 
            distances_to_left_hand, 
            distances_to_right_elbow, 
            distances_to_right_hand, 
            distances_to_left_knee, 
            distances_to_left_foot, 
            distances_to_right_knee, 
            distances_to_right_foot
        ]

        angles_list = [
            angles_head_to_left_elbow, 
            angles_left_elbow_to_left_hand, 
            angles_left_hand_to_left_knee, 
            angles_left_knee_to_left_foot, 
            angles_left_foot_to_right_foot, 
            angles_right_foot_to_right_knee, 
            angles_right_knee_to_right_hand, 
            angles_right_hand_to_right_elbow, 
            angles_right_elbow_to_head
        ]

        bins_distances = [freedman_diaconis_bins(distances) for distances in distances_list]
        bins_angles = [freedman_diaconis_bins(angles) for angles in angles_list]

        if ONE_ITER:
            # Plot normalized histograms for distances
            fig, axes = plt.subplots(3, 3, figsize=(9.5, 9.5))
            axes = axes.flatten()
            labels = ["Head", "Left Elbow", "Left Hand", "Right Elbow", "Right Hand", "Left Knee", "Left Foot", "Right Knee", "Right Foot"]
            distances_list = [distances_to_head, distances_to_left_elbow, distances_to_left_hand, distances_to_right_elbow, distances_to_right_hand, distances_to_left_knee, distances_to_left_foot, distances_to_right_knee, distances_to_right_foot]

            for i, (distances, label, bins) in enumerate(zip(distances_list, labels, bins_distances)):
                axes[i].hist(distances, bins=bins, weights=weights, alpha=0.5, label=label)
                axes[i].set_xlabel("Distance")
                axes[i].set_ylabel("Normalized Frequency")
                axes[i].set_title(f"Normalized Histogram of {label} Distances", fontsize=8)

            plt.tight_layout()
            plt.show()

            # Plot normalized histograms for angles
            fig, axes = plt.subplots(3, 3, figsize=(9.5, 9.5))
            axes = axes.flatten()
            labels = ["Head-Left Elbow", "Left Elbow-Left Hand", "Left Hand-Left Knee", "Left Knee-Left Foot", "Left Foot-Right Foot", "Right Foot-Right Knee", "Right Knee-Right Hand", "Right Hand-Right Elbow", "Right Elbow-Head"]
            angles_list = [angles_head_to_left_elbow, angles_left_elbow_to_left_hand, angles_left_hand_to_left_knee, angles_left_knee_to_left_foot, angles_left_foot_to_right_foot, angles_right_foot_to_right_knee, angles_right_knee_to_right_hand, angles_right_hand_to_right_elbow, angles_right_elbow_to_head]

            for i, (angles, label, bins) in enumerate(zip(angles_list, labels, bins_angles)):
                axes[i].hist(angles, bins=bins, weights=weights, alpha=0.5, label=label)
                axes[i].set_xlabel("Angle")
                axes[i].set_ylabel("Normalized Frequency")
                axes[i].set_title(f"Normalized Histogram of {label} Angles", fontsize=7)

            plt.tight_layout()
            plt.show()
            # If only ONE_ITER, then return after 1st iteration
            return
        
        concatenated_histograms_str = " ".join([str(val) for val in concatenated_histograms])
        print(concatenated_histograms)
        with open(filepath, "a") as f:
            # Append concatenated_histograms to file
            f.write(concatenated_histograms_str + " " + str(activity) + "\n")

def main():

    ONE_ITER = False # set True to get plots, False to do file output
    TESTING_SET = False # toggle for training or testing
    RAD_ALGORITHM = False # set False for custom, True for RAD
    
    source = "test" if TESTING_SET else "train"
    data = []
    activities = []
    
    for filename in os.listdir(FOLDER_PATH + source):
        filename = os.path.join(FOLDER_PATH, source, filename)
        file_data, activity = load_from_file(filename)
        activities.append(activity)
        data.append(file_data)
    
    if TESTING_SET:
        if RAD_ALGORITHM:
            RAD_representation(data, activities, testing=True, ONE_ITER=ONE_ITER)
        else:
            custom_representation(data, activities, testing=True, ONE_ITER=ONE_ITER)
    else:
        if RAD_ALGORITHM:
            RAD_representation(data, activities, testing=False, ONE_ITER=ONE_ITER)
        else:
            custom_representation(data, activities, testing=False, ONE_ITER=ONE_ITER)

if __name__ == "__main__":
	main()