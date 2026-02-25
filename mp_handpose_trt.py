import numpy as np
import cv2 as cv
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorrt")
warnings.filterwarnings("ignore", category=UserWarning, module="tensorrt")

class MPHandPoseTRT:
    def __init__(self, modelPath, confThreshold=0.8, use_gpu=True):
        self.model_path = modelPath
        self.conf_threshold = confThreshold
        self.use_gpu = use_gpu

        self.input_size = np.array([224, 224])  # wh
        self.PALM_LANDMARK_IDS = [0, 5, 9, 13, 17, 1, 2]
        self.PALM_LANDMARKS_INDEX_OF_PALM_BASE = 0
        self.PALM_LANDMARKS_INDEX_OF_MIDDLE_FINGER_BASE = 2
        self.PALM_BOX_PRE_SHIFT_VECTOR = [0, 0]
        self.PALM_BOX_PRE_ENLARGE_FACTOR = 4
        self.PALM_BOX_SHIFT_VECTOR = [0, -0.4]
        self.PALM_BOX_ENLARGE_FACTOR = 3
        self.HAND_BOX_SHIFT_VECTOR = [0, -0.1]
        self.HAND_BOX_ENLARGE_FACTOR = 1.65

        # 加载 TensorRT 引擎
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(modelPath, 'rb') as f:
            self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # 获取输入输出信息
        self.input_name = self.engine.get_binding_name(0)
        
        # 获取输出名称和顺序
        self.output_names = []
        self.output_shapes = {}
        for i in range(1, self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            self.output_names.append(name)
            self.output_shapes[name] = self.engine.get_binding_shape(i)
        
        # 打印输出信息用于调试
        print(f"TensorRT引擎输出顺序:")
        for i, name in enumerate(self.output_names):
            print(f"  {i}: {name}, 形状={self.output_shapes[name]}")

        # 分配内存
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()

        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):
                self.inputs.append(host_mem)
                self.inputs.append(device_mem)
            else:
                self.outputs.append(host_mem)
                self.outputs.append(device_mem)

    @property
    def name(self):
        return self.__class__.__name__

    def _cropAndPadFromPalm(self, image, palm_bbox, for_rotation=False):
        wh_palm_bbox = palm_bbox[1] - palm_bbox[0]
        if for_rotation:
            shift_vector = self.PALM_BOX_PRE_SHIFT_VECTOR
        else:
            shift_vector = self.PALM_BOX_SHIFT_VECTOR
        shift_vector = shift_vector * wh_palm_bbox
        palm_bbox = palm_bbox + shift_vector
        center_palm_bbox = np.sum(palm_bbox, axis=0) / 2
        wh_palm_bbox = palm_bbox[1] - palm_bbox[0]
        if for_rotation:
            enlarge_scale = self.PALM_BOX_PRE_ENLARGE_FACTOR
        else:
            enlarge_scale = self.PALM_BOX_ENLARGE_FACTOR
        new_half_size = wh_palm_bbox * enlarge_scale / 2
        palm_bbox = np.array([
            center_palm_bbox - new_half_size,
            center_palm_bbox + new_half_size])
        palm_bbox = palm_bbox.astype(np.int32)
        palm_bbox[:, 0] = np.clip(palm_bbox[:, 0], 0, image.shape[1])
        palm_bbox[:, 1] = np.clip(palm_bbox[:, 1], 0, image.shape[0])
        image = image[palm_bbox[0][1]:palm_bbox[1][1], palm_bbox[0][0]:palm_bbox[1][0], :]
        
        if image.size == 0:
            return None, palm_bbox.astype(np.float32), np.array([0, 0], dtype=np.int32)
        
        if for_rotation:
            side_len = np.linalg.norm(image.shape[:2])
        else:
            side_len = max(image.shape[0], image.shape[1])

        side_len = int(side_len)
        pad_h = side_len - image.shape[0]
        pad_w = side_len - image.shape[1]
        left = pad_w // 2
        top = pad_h // 2
        right = pad_w - left
        bottom = pad_h - top
        image = cv.copyMakeBorder(image, top, bottom, left, right, cv.BORDER_CONSTANT, None, (0, 0, 0))
        bias = palm_bbox[0] - [left, top]
        return image, palm_bbox.astype(np.float32), bias

    def _preprocess(self, image, palm):
        pad_bias = np.array([0, 0], dtype=np.int32)
        palm_bbox_orig = palm[0:4].reshape(2, 2).copy()
        
        image, palm_bbox, bias = self._cropAndPadFromPalm(image, palm_bbox_orig, True)
        
        if image is None:
            return None, None, None, None, None
            
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        pad_bias += bias

        palm_bbox -= pad_bias
        
        palm_landmarks = palm[4:18].reshape(7, 2)
        palm_landmarks = palm_bbox_orig[0] + palm_landmarks - pad_bias
        p1 = palm_landmarks[self.PALM_LANDMARKS_INDEX_OF_PALM_BASE]
        p2 = palm_landmarks[self.PALM_LANDMARKS_INDEX_OF_MIDDLE_FINGER_BASE]
        radians = np.pi / 2 - np.arctan2(-(p2[1] - p1[1]), p2[0] - p1[0])
        radians = radians - 2 * np.pi * np.floor((radians + np.pi) / (2 * np.pi))
        angle = np.rad2deg(radians)
        center_palm_bbox = np.sum(palm_bbox, axis=0) / 2
        rotation_matrix = cv.getRotationMatrix2D(center_palm_bbox, angle, 1.0)
        rotated_image = cv.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
        
        homogeneous_coord = np.c_[palm_landmarks, np.ones(palm_landmarks.shape[0])]
        rotated_palm_landmarks = np.array([
            np.dot(homogeneous_coord, rotation_matrix[0]),
            np.dot(homogeneous_coord, rotation_matrix[1])])
        rotated_palm_bbox = np.array([
            np.amin(rotated_palm_landmarks, axis=1),
            np.amax(rotated_palm_landmarks, axis=1)])

        crop, rotated_palm_bbox, _ = self._cropAndPadFromPalm(rotated_image, rotated_palm_bbox)
        if crop is None or crop.size == 0:
            return None, None, None, None, None
        
        blob = cv.resize(crop, dsize=self.input_size, interpolation=cv.INTER_AREA).astype(np.float32)
        blob = blob / 255.
        blob = np.transpose(blob, (2, 0, 1))

        return blob[np.newaxis, :, :, :], rotated_palm_bbox, angle, rotation_matrix, pad_bias

    def infer(self, image, palm):
        input_blob, rotated_palm_bbox, angle, rotation_matrix, pad_bias = self._preprocess(image, palm)
        if input_blob is None:
            return None

        np.copyto(self.inputs[0], input_blob.ravel())
        cuda.memcpy_htod_async(self.inputs[1], self.inputs[0], self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        
        output_count = 0
        for i in range(self.engine.num_bindings):
            if not self.engine.binding_is_input(i):
                host_idx = output_count * 2
                device_idx = output_count * 2 + 1
                cuda.memcpy_dtoh_async(self.outputs[host_idx], self.outputs[device_idx], self.stream)
                output_count += 1
        
        self.stream.synchronize()

        # 按名称解析输出
        output_dict = {}
        output_count = 0
        for i in range(self.engine.num_bindings):
            if not self.engine.binding_is_input(i):
                name = self.output_names[output_count]
                output_shape = self.output_shapes[name]
                output_size = trt.volume(output_shape)
                output_index = output_count * 2
                output_data = self.outputs[output_index][:output_size].reshape(output_shape)
                output_dict[name] = output_data
                output_count += 1

        results = self._postprocess(output_dict, rotated_palm_bbox, angle, rotation_matrix, pad_bias)
        return results

    def _postprocess(self, output_dict, rotated_palm_bbox, angle, rotation_matrix, pad_bias):
        # TensorRT输出顺序: Identity_2(handedness), Identity_1(confidence), Identity(landmarks), Identity_3(world_landmarks)
        # 按名称获取输出
        landmarks = output_dict['Identity']
        conf = output_dict['Identity_1']
        handedness = output_dict['Identity_2']
        landmarks_world = output_dict['Identity_3']

        conf = conf[0][0]
        if conf < self.conf_threshold:
            return None

        landmarks = landmarks[0].reshape(-1, 3)
        landmarks_world = landmarks_world[0].reshape(-1, 3)

        wh_rotated_palm_bbox = rotated_palm_bbox[1] - rotated_palm_bbox[0]
        scale_factor = wh_rotated_palm_bbox / self.input_size
        landmarks[:, :2] = (landmarks[:, :2] - self.input_size / 2) * max(scale_factor)
        landmarks[:, 2] = landmarks[:, 2] * max(scale_factor)
        coords_rotation_matrix = cv.getRotationMatrix2D((0, 0), angle, 1.0)
        rotated_landmarks = np.dot(landmarks[:, :2], coords_rotation_matrix[:, :2])
        rotated_landmarks = np.c_[rotated_landmarks, landmarks[:, 2]]
        rotated_landmarks_world = np.dot(landmarks_world[:, :2], coords_rotation_matrix[:, :2])
        rotated_landmarks_world = np.c_[rotated_landmarks_world, landmarks_world[:, 2]]
        
        rotation_component = np.array([
            [rotation_matrix[0][0], rotation_matrix[1][0]],
            [rotation_matrix[0][1], rotation_matrix[1][1]]])
        translation_component = np.array([
            rotation_matrix[0][2], rotation_matrix[1][2]])
        inverted_translation = np.array([
            -np.dot(rotation_component[0], translation_component),
            -np.dot(rotation_component[1], translation_component)])
        inverse_rotation_matrix = np.c_[rotation_component, inverted_translation]
        center = np.append(np.sum(rotated_palm_bbox, axis=0) / 2, 1)
        original_center = np.array([
            np.dot(center, inverse_rotation_matrix[0]),
            np.dot(center, inverse_rotation_matrix[1])])
        landmarks[:, :2] = rotated_landmarks[:, :2] + original_center + pad_bias

        bbox = np.array([
            np.amin(landmarks[:, :2], axis=0),
            np.amax(landmarks[:, :2], axis=0)])
        wh_bbox = bbox[1] - bbox[0]
        shift_vector = self.HAND_BOX_SHIFT_VECTOR * wh_bbox
        bbox = bbox + shift_vector
        center_bbox = np.sum(bbox, axis=0) / 2
        wh_bbox = bbox[1] - bbox[0]
        new_half_size = wh_bbox * self.HAND_BOX_ENLARGE_FACTOR / 2
        bbox = np.array([
            center_bbox - new_half_size,
            center_bbox + new_half_size])

        return np.r_[bbox.reshape(-1), landmarks.reshape(-1), rotated_landmarks_world.reshape(-1), handedness[0][0], conf]