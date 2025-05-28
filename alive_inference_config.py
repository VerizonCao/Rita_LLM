import json
import os
import tempfile


class AliveInferenceConfig:
    def __init__(self):
        # basic param
        self.seconds_per_window = 3
        self.vid_fps = 25

        # input io
        io_file_path = self.check_file_paths()
        self.io_init(io_file_path)

        # audio2motion inference param
        self.a2m_cfg = 0.5
        self.a2m_mouth_ratio = 0.35
        self.a2m_step = 1
        # a2m smooth
        self.a2m_R_exp, self.a2m_Q_exp = 0.05, 0.5
        self.a2m_R_pose, self.a2m_Q_pose = 0.3, 0.05
        # a2m gaze
        self.a2m_gaze_adjust_opts = (0.7, 2, 0.5, 0.25, -5, 2.5, -20.5)

        # a2m model arch related
        self.prev_frame_count = 10
        self.gen_frame_count = 65  # 3 * 25 - 10
        self.headpose_bound_list = [
            -21,
            25,
            -30,
            30,
            -23,
            23,
            -0.3,
            0.3,
            -0.3,
            0.28,
        ]
        self.gaze_index_list = [4, 6, 7, 33, 34, 40, 45, 46, 48]

    def io_init(self, file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
            self.io_source_portrait_path = data["io_source_portrait_path"]
            self.io_source_portrait_folder = data["io_source_portrait_folder"]
            # LLM
            self.llm_user_nickname = (
                data["llm_user_nickname"] if "llm_user_nickname" in data else ""
            )
            self.llm_user_bio = data["llm_user_bio"] if "llm_user_bio" in data else ""
            self.llm_assistant_nickname = (
                data["llm_assistant_nickname"]
                if "llm_assistant_nickname" in data
                else ""
            )
            self.llm_assistant_bio = (
                data["llm_assistant_bio"] if "llm_assistant_bio" in data else ""
            )
            self.llm_assistant_additional_characteristics = (
                data["llm_assistant_additional_characteristics"]
                if "llm_assistant_additional_characteristics" in data
                else ""
            )
            self.llm_conversation_context = (
                data["llm_conversation_context"]
                if "llm_conversation_context" in data
                else ""
            )
            self.llm_expression_list = ["neutral-neutral"]
            # TTS
            self.tts_voice_id_cartesia = (
                data["tts_voice_id_cartesia"]
                if "tts_voice_id_cartesia" in data
                else "78ab82d5-25be-4f7d-82b3-7ad64e5b85b2"
            )
            # A2M
            self.io_audio_path = data["io_audio_path"]
            self.io_a2m_config_path = data["io_a2m_config_path"]
            self.io_a2m_weight_path = data["io_a2m_weight_path"]
            self.gaze_ref_img_path = data["gaze_ref_img_path"]
            self.faster_config_path = data["faster_config_path"]
            self.text_motion_pair = data["text_motion_pair"]

    def check_file_paths(self):
        current_dir = os.path.dirname(__file__)
        file_path = os.path.join(current_dir, "file_paths.json")
        assert os.path.exists(
            file_path
        ), "file_paths.json does not exist in the current directory"
        return file_path

    def replace_base_serve_image(self, avatarSource: str):
        current_dir = os.path.dirname(__file__)
        file_path = os.path.join(current_dir, "file_paths.json")
        assert os.path.exists(
            file_path
        ), "file_paths.json does not exist in the current directory"

        # Read the current JSON file
        with open(file_path, "r") as f:
            data = json.load(f)

        # Update the io_source_portrait_path
        old_serve_path = data["io_source_portrait_path"]
        data["io_source_portrait_path"] = avatarSource

        # Write the updated JSON back to the file
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)

        return old_serve_path

    def get_llm_tuple(self):
        llm_data = (
            self.llm_user_nickname,
            self.llm_user_bio,
            self.llm_assistant_nickname,
            self.llm_assistant_bio,
            self.llm_assistant_additional_characteristics,
            self.llm_conversation_context,
            self.llm_expression_list,
        )
        return llm_data

    def replace_all_llm(self, llm_configs, io_source_portrait_folder_reset=None, avatar_id=None):
        current_dir = os.path.dirname(__file__)
        file_path = os.path.join(current_dir, "file_paths.json")
        assert os.path.exists(
            file_path
        ), "file_paths.json does not exist in the current directory"

        # Read the current JSON file
        with open(file_path, "r") as f:
            data = json.load(f)

        # Extract only LLM-related fields
        old_llm_configs = {
            "llm_user_nickname": data["llm_user_nickname"],
            "llm_user_bio": data["llm_user_bio"],
            "llm_assistant_nickname": data["llm_assistant_nickname"],
            "llm_assistant_bio": data["llm_assistant_bio"],
            "llm_assistant_additional_characteristics": data[
                "llm_assistant_additional_characteristics"
            ],
            "llm_conversation_context": data["llm_conversation_context"],
            "tts_voice_id_cartesia": data["tts_voice_id_cartesia"],
        }

        io_source_portrait_folder_old = data["io_source_portrait_folder"]
        if not io_source_portrait_folder_reset:
            temp_dir = tempfile.gettempdir()
            if avatar_id and avatar_id != "":
                io_source_portrait_folder_new = os.path.join(temp_dir, avatar_id)
            else:
                io_source_portrait_folder_new = os.path.join(temp_dir, io_source_portrait_folder_old)


        # Update the LLM fields with new values only if they are not empty and different from current value
        for field in old_llm_configs.keys():
            if (
                field in llm_configs
                and llm_configs[field]
                and llm_configs[field].strip()
            ):
                new_value = llm_configs[field].strip()
                if new_value != old_llm_configs[field]:
                    print(f"Updating {field} to {new_value}")
                    data[field] = new_value

        if io_source_portrait_folder_reset:
            data["io_source_portrait_folder"] = io_source_portrait_folder_reset
        else:
            data["io_source_portrait_folder"] = io_source_portrait_folder_new

        # Write the updated JSON back to the file
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)

        return old_llm_configs, io_source_portrait_folder_old
