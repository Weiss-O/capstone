
camera_settings = {
    "resolution": [4608, 2592],
    "orientation": "portrait"
}
controller_settings = {
    "baudrate": 9600,
    "port": "/dev/ttyACM0",
    "timeout": 1
}
server_settings = {
    "HOST": "100.72.50.30",
    "PORT": 5000
}
sam2_model = {
    "config_path": "configs/sam2.1/sam2.1_hiera_l.yaml",
    "checkpoint_path": "checkpoints/sam2.1_hiera_large.pt"
}
idle_time_vacant = 5,
idle_time_occupied = 5,
baseline = {
    "POS1":{
        "image": "baseline/baseline1.jpg",
        "camera_pos": [45, 60]
    },
    "POS2":{
        "image": "baseline/baseline2.jpg",
        "camera_pos": [80, 39]
    },
    "POS3":{
        "image": "baseline/baseline3.jpg",
        "camera_pos": [45, 39]
    },
    "POS4":{
        "image": "baseline/baseline4.jpg",
        "camera_pos": [10, 39]
    }
}