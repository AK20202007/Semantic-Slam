import os
import pickle

class CloudSync:
    def __init__(self, bucket_name="semantic-slam-maps", mock=True):
        self.bucket_name = bucket_name
        self.mock = mock
        
        if self.mock:
            self.local_dir = os.path.join(os.path.dirname(__file__), "..", "mock_cloud_storage")
            os.makedirs(self.local_dir, exist_ok=True)
            print(f"Initialized mock cloud storage at {self.local_dir}")
        else:
            import boto3
            self.s3 = boto3.client('s3')
            print(f"Initialized AWS S3 client for bucket {self.bucket_name}")
            
    def save_map(self, map_name, point_cloud, labels_3d, trajectory):
        print(f"Saving map '{map_name}' to cloud...")
        
        data = {
            'point_cloud': point_cloud,
            'labels_3d': labels_3d,
            'trajectory': trajectory
        }
        
        if self.mock:
            filepath = os.path.join(self.local_dir, f"{map_name}.pkl")
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            print(f"Map successfully saved locally to {filepath}")
        else:
            pickle_data = pickle.dumps(data)
            self.s3.put_object(Bucket=self.bucket_name, Key=f"{map_name}.pkl", Body=pickle_data)
            print(f"Map successfully uploaded to S3 bucket {self.bucket_name}")

    def load_map(self, map_name):
        print(f"Loading map '{map_name}' from cloud...")
        
        if self.mock:
            filepath = os.path.join(self.local_dir, f"{map_name}.pkl")
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                print("Map loaded successfully.")
                return data['point_cloud'], data['labels_3d'], data['trajectory']
            else:
                print("Map not found.")
                return [], [], []
        else:
            try:
                response = self.s3.get_object(Bucket=self.bucket_name, Key=f"{map_name}.pkl")
                data = pickle.loads(response['Body'].read())
                print("Map loaded successfully from S3.")
                return data['point_cloud'], data['labels_3d'], data['trajectory']
            except Exception as e:
                print(f"Failed to load map from S3: {e}")
                return [], [], []
