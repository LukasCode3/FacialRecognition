# Celebrity Face Recognition System

Detects and identifies celebrities in group photos using DeepFace and OpenCV.

## Installation

```bash
pip install -r requirements.txt
```

## Setup

Create celebrity database:
```
celebrity_database/
├── Celebrity_Name_1/
│   ├── image1.jpg
│   └── image2.jpg
└── Celebrity_Name_2/
    └── image1.jpg
```

## Usage

```python
from celebrity_face_recognition import CelebrityRecognizer

recognizer = CelebrityRecognizer(
    celebrity_db_path='./celebrity_database',
    model_name='VGG-Face',
    threshold=0.4
)

results = recognizer.process_image('photo.jpg', visualize=True)
print(f"Found {results['celebrity_count']} celebrities")
```

## Configuration

**Models**: VGG-Face (fast), Facenet (balanced), Facenet512 (accurate), OpenFace (fastest)

**Threshold**: Lower = stricter matching (0.3-0.5 recommended)

**Distance Metrics**: cosine (recommended), euclidean, euclidean_l2

## Output Format

```python
{
    'total_faces': 5,
    'celebrity_count': 2,
    'celebrities_found': [
        {
            'name': 'Celebrity_Name',
            'confidence': 0.87,
            'bounding_box': {'x': 100, 'y': 150, 'w': 120, 'h': 150}
        }
    ]
}
```
## License

Respect privacy laws and dataset licenses. For research and educational use only.