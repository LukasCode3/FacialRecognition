import cv2
import numpy as np
from deepface import DeepFace
import os
from pathlib import Path
import pickle
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CelebrityRecognizer:
    """
    A class to recognize celebrities in images using facial recognition.
    """
    
    def __init__(self, celebrity_db_path: str, model_name: str = "VGG-Face", 
                 distance_metric: str = "cosine", threshold: float = 0.4):
        """
        Initialize the Celebrity Recognizer.
        
        Args:
            celebrity_db_path: Path to celebrity face database directory
            model_name: DeepFace model to use (VGG-Face, Facenet, OpenFace, DeepFace, etc.)
            distance_metric: Distance metric (cosine, euclidean, euclidean_l2)
            threshold: Recognition threshold (lower = stricter matching)
        """
        self.celebrity_db_path = celebrity_db_path
        self.model_name = model_name
        self.distance_metric = distance_metric
        self.threshold = threshold
        self.celebrity_embeddings = {}
        self.load_celebrity_database()
    
    def load_celebrity_database(self):
        """
        Load celebrity face embeddings from the database directory.
        Expected structure:
        celebrity_db/
            ├── celebrity_name_1/
            │   ├── image1.jpg
            │   ├── image2.jpg
            └── celebrity_name_2/
                ├── image1.jpg
                └── image2.jpg
        """
        logger.info(f"Loading celebrity database from {self.celebrity_db_path}")
        
        if not os.path.exists(self.celebrity_db_path):
            raise FileNotFoundError(f"Celebrity database path not found: {self.celebrity_db_path}")
        
        # Check for pre-computed embeddings
        embeddings_file = os.path.join(self.celebrity_db_path, "embeddings.pkl")
        
        if os.path.exists(embeddings_file):
            logger.info("Loading pre-computed embeddings...")
            with open(embeddings_file, 'rb') as f:
                self.celebrity_embeddings = pickle.load(f)
            logger.info(f"Loaded {len(self.celebrity_embeddings)} celebrity profiles")
            return
        
        # Compute embeddings from scratch
        celebrity_dirs = [d for d in Path(self.celebrity_db_path).iterdir() if d.is_dir()]
        
        for celeb_dir in celebrity_dirs:
            celeb_name = celeb_dir.name
            embeddings = []
            
            image_files = list(celeb_dir.glob('*.jpg')) + list(celeb_dir.glob('*.png')) + \
                         list(celeb_dir.glob('*.jpeg'))
            
            for img_path in image_files:
                try:
                    # Extract embedding for this celebrity image
                    embedding = DeepFace.represent(
                        img_path=str(img_path),
                        model_name=self.model_name,
                        enforce_detection=False
                    )
                    
                    if embedding:
                        embeddings.append(embedding[0]["embedding"])
                        
                except Exception as e:
                    logger.warning(f"Error processing {img_path}: {e}")
                    continue
            
            if embeddings:
                # Store average embedding for this celebrity
                self.celebrity_embeddings[celeb_name] = {
                    'embeddings': embeddings,
                    'avg_embedding': np.mean(embeddings, axis=0).tolist()
                }
                logger.info(f"Loaded {len(embeddings)} images for {celeb_name}")
        
        # Save embeddings for future use
        with open(embeddings_file, 'wb') as f:
            pickle.dump(self.celebrity_embeddings, f)
        
        logger.info(f"Celebrity database loaded: {len(self.celebrity_embeddings)} celebrities")
    
    def detect_faces(self, image_path: str) -> List[Dict]:
        """
        Detect all faces in the input image.
        
        Args:
            image_path: Path to input image
            
        Returns:
            List of face detections with bounding boxes and embeddings
        """
        try:
            # Use DeepFace to extract faces and embeddings
            faces = DeepFace.extract_faces(
                img_path=image_path,
                detector_backend='opencv',  # Can also use: retinaface, mtcnn, ssd
                enforce_detection=False
            )
            
            # Get embeddings for each detected face
            face_data = []
            for i, face in enumerate(faces):
                try:
                    embedding = DeepFace.represent(
                        img_path=image_path,
                        model_name=self.model_name,
                        enforce_detection=False
                    )
                    
                    if embedding and i < len(embedding):
                        face_data.append({
                            'face': face['face'],
                            'facial_area': face['facial_area'],
                            'embedding': embedding[i]['embedding'],
                            'confidence': face['confidence']
                        })
                except Exception as e:
                    logger.warning(f"Error getting embedding for face {i}: {e}")
                    
            return face_data
            
        except Exception as e:
            logger.error(f"Error detecting faces: {e}")
            return []
    
    def calculate_distance(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate distance between two face embeddings.
        
        Args:
            embedding1: First face embedding
            embedding2: Second face embedding
            
        Returns:
            Distance value
        """
        emb1 = np.array(embedding1)
        emb2 = np.array(embedding2)
        
        if self.distance_metric == 'cosine':
            return 1 - np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        elif self.distance_metric == 'euclidean':
            return np.linalg.norm(emb1 - emb2)
        elif self.distance_metric == 'euclidean_l2':
            return np.linalg.norm(emb1 - emb2) / np.sqrt(len(emb1))
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
    
    def identify_celebrity(self, face_embedding: List[float]) -> Tuple[str, float]:
        """
        Identify if a face belongs to a celebrity.
        
        Args:
            face_embedding: Face embedding to match
            
        Returns:
            Tuple of (celebrity_name, confidence_score) or (None, None) if no match
        """
        best_match = None
        best_distance = float('inf')
        
        for celeb_name, celeb_data in self.celebrity_embeddings.items():
            # Compare with average embedding
            distance = self.calculate_distance(
                face_embedding,
                celeb_data['avg_embedding']
            )
            
            # Also check against individual embeddings for better accuracy
            for individual_embedding in celeb_data['embeddings']:
                ind_distance = self.calculate_distance(face_embedding, individual_embedding)
                distance = min(distance, ind_distance)
            
            if distance < best_distance:
                best_distance = distance
                best_match = celeb_name
        
        # Check if best match is below threshold
        if best_distance < self.threshold:
            confidence = 1 - best_distance  # Convert distance to confidence
            return best_match, confidence
        
        return None, None
    
    def process_image(self, image_path: str, visualize: bool = True) -> Dict:
        """
        Process an image to detect and identify celebrities.
        
        Args:
            image_path: Path to input image
            visualize: Whether to create annotated output image
            
        Returns:
            Dictionary with results
        """
        logger.info(f"Processing image: {image_path}")
        
        # Detect all faces
        faces = self.detect_faces(image_path)
        logger.info(f"Detected {len(faces)} faces")
        
        # Identify celebrities
        results = {
            'total_faces': len(faces),
            'celebrities_found': [],
            'celebrity_count': 0,
            'image_path': image_path
        }
        
        for i, face_data in enumerate(faces):
            celeb_name, confidence = self.identify_celebrity(face_data['embedding'])
            
            if celeb_name:
                results['celebrities_found'].append({
                    'name': celeb_name,
                    'confidence': float(confidence),
                    'face_index': i,
                    'bounding_box': face_data['facial_area']
                })
                logger.info(f"Celebrity detected: {celeb_name} (confidence: {confidence:.2f})")
        
        results['celebrity_count'] = len(results['celebrities_found'])
        
        # Visualize results if requested
        if visualize and results['celebrity_count'] > 0:
            self.visualize_results(image_path, results)
        
        return results
    
    def visualize_results(self, image_path: str, results: Dict, 
                         output_path: str = None):
        """
        Create annotated image with celebrity detections.
        
        Args:
            image_path: Path to input image
            results: Results dictionary from process_image
            output_path: Path to save annotated image
        """
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Could not read image: {image_path}")
            return
        
        # Draw bounding boxes and labels for celebrities
        for celeb in results['celebrities_found']:
            bbox = celeb['bounding_box']
            name = celeb['name']
            confidence = celeb['confidence']
            
            # Draw rectangle
            cv2.rectangle(
                img,
                (bbox['x'], bbox['y']),
                (bbox['x'] + bbox['w'], bbox['y'] + bbox['h']),
                (0, 255, 0),
                0
            )
            
            # Draw label background
            label = f"{name} ({confidence:.2f})"
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            cv2.rectangle(
                img,
                (bbox['x'], bbox['y'] - label_h - 10),
                (bbox['x'] + label_w, bbox['y']),
                (0, 255, 0),
                0
            )
            
            # Draw label text
            cv2.putText(
                img,
                label,
                (bbox['x'], bbox['y'] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )
        
        # Save annotated image
        if output_path is None:
            output_path = image_path.replace('.', '_annotated.')
        
        cv2.imwrite(output_path, img)
        logger.info(f"Annotated image saved to: {output_path}")
        
        return output_path


def main():
    """
    Example usage of the Celebrity Recognizer
    """
    # Initialize recognizer
    recognizer = CelebrityRecognizer(
        celebrity_db_path='./celebrity_database',
        model_name='VGG-Face',
        distance_metric='cosine',
        threshold=0.4
    )
    
    # Process an image
    image_path = 'test.jpg'
    results = recognizer.process_image(image_path, visualize=True)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Celebrity Recognition Results")
    print(f"{'='*60}")
    print(f"Total faces detected: {results['total_faces']}")
    print(f"Celebrities found: {results['celebrity_count']}")
    print(f"\nDetailed results:")
    
    for celeb in results['celebrities_found']:
        print(f"  - {celeb['name']} (confidence: {celeb['confidence']:.2%})")
    
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
