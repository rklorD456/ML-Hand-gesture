import pickle
from helper import RealTimeGesturePredictor


if __name__ == "__main__":
    
    # Load the trained ML Model
    with open('final_model.pkl', 'rb') as f:
        model = pickle.load(f)
        
    # class names in the order your model expects them
    hagrid_classes = ['call', 'dislike','fist','four','like','mute',
                      'ok','one','palm','peace','peace_inverted','rock',
                      'stop','stop_inverted','three',
                      'three2','two_up','two_up_inverted']
                      
    # Initialize and start the live webcam stream
    predictor = RealTimeGesturePredictor(
        model=model, 
        class_names=hagrid_classes,
        model_asset_path='models\\hand_landmarker.task'
    )
    
    predictor.start_stream(video_source=1)