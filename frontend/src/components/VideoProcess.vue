<!-- VideoProcess.vue 视频处理页面 -->
<template>
  <div class="container">
    <el-upload
      class="upload-video"
      action="/api/upload-video"
      :data="{ asr_mode: asrMode }"
      :on-success="handleUploadSuccess"
      :on-error="handleUploadError"
      accept="video/*"
      :show-file-list="true"
    >
      <el-button type="primary">上传视频</el-button>
    </el-upload>
    
    <el-radio-group v-model="asrMode" class="asr-select">
      <el-radio label="hubert">Hubert（效果好，25fps）</el-radio>
      <el-radio label="wenet">Wenet（速度快，20fps）</el-radio>
    </el-radio-group>
    
    <el-progress v-if="processing" :percentage="processPercent" stroke-width="4"></el-progress>
    
    <el-card v-if="validationResult">
      <div slot="header">视频校验结果</div>
      <div v-if="validationResult.valid" class="success">
        <el-icon><check /></el-icon> 视频校验通过，可进行下一步预处理
      </div>
      <div v-else class="error">
        <el-icon><close /></el-icon> 校验失败：
        <ul>
          <li v-for="err in validationResult.errors" :key="err">{{ err }}</li>
        </ul>
      </div>
    </el-card>
    
    <el-button 
      v-if="showPreprocessBtn" 
      type="success" 
      @click="startPreprocess"
    >
      开始数据预处理
    </el-button>
  </div>
</template>

<script setup>
import { ref } from 'vue';
import { Check, Close } from '@element-plus/icons-vue';
import axios from 'axios';

const asrMode = ref('hubert');
const processing = ref(false);
const processPercent = ref(0);
const validationResult = ref(null);
const taskId = ref('');
const showPreprocessBtn = ref(false);

const handleUploadSuccess = (response) => {
  taskId.value = response.task_id;
  validationResult.value = response.validation;
  if (response.success) {
    showPreprocessBtn.value = true;
  }
};

const handleUploadError = (error) => {
  console.error('上传失败', error);
};

const startPreprocess = async () => {
  processing.value = true;
  processPercent.value = 30;
  try {
    await axios.post('/api/preprocess', {
      task_id: taskId.value,
      asr_mode: asrMode.value
    });
    processPercent.value = 100;
    // 跳转至训练页面
  } catch (e) {
    console.error('预处理失败', e);
  } finally {
    processing.value = false;
  }
};
</script>