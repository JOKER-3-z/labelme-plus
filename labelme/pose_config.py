

POSE_TEMPLATE_JSON = "pose_template.json"

pose_define = {
"visible_threshold":0.5,
"keypoints": ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",\
              "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", \
              "right_knee", "left_ankle", "right_ankle"],
"keypoints_priority":["nose", "left_ear", "right_ear", "left_shoulder", "right_shoulder",\
              "left_hip", "right_hip"],
"skeleton":[[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], \
            [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]],
"location": {"nose": [0.48337208483055016, 0.08411035446132355], "left_eye": [0.5168392869724511, 0.06839532041208311], \
             "right_eye": [0.44814345099697017, 0.06839532041208311], "left_ear": [0.5450221940393151, 0.09546010127466387], \
             "right_ear": [0.4111533854717114, 0.08847564169722366], "left_shoulder": [0.6718452758402028, 0.19586170770036665],\
             "right_shoulder": [0.3089903473543295, 0.19324253535882657], "left_elbow": [0.7317339533572886, 0.35999650777021125],\
             "right_elbow": [0.24029451137884864, 0.3608695652173913], "left_wrist": [0.8286126963996334, 0.492701239741575], \
             "right_wrist": [0.13108574649475088, 0.4970665269774751], "left_hip": [0.583773691256253, 0.4289680460974332], \
             "right_hip": [0.37064045656309436, 0.4307141609917932]}
}