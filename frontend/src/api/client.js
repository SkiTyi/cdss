import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 60000,
})

export const documents = {
  list: (params) => api.get('/documents', { params }),
  stats: () => api.get('/documents/stats'),
  load: () => api.post('/documents/load'),
  get: (id) => api.get(`/documents/${id}`),
  delete: (id) => api.delete(`/documents/${id}`),
  deleteBulk: (ids) => api.post('/documents/delete-bulk', { ids }),
  deleteShort: (threshold, type) =>
    api.post('/documents/delete-short', { threshold, type: type || null }),
  reset: () => api.post('/documents/reset'),
}

export const extraction = {
  listJobs: () => api.get('/extraction/jobs'),
  createJob: (data) => api.post('/extraction/jobs', data),
  getJob: (id) => api.get(`/extraction/jobs/${id}`),
  cancelJob: (id) => api.post(`/extraction/jobs/${id}/cancel`),
  restartJob: (id) => api.post(`/extraction/jobs/${id}/restart`),
  deleteJob: (id) => api.delete(`/extraction/jobs/${id}`),
  listInstances: (params) => api.get('/extraction/instances', { params }),
  approveInstance: (id) => api.patch(`/extraction/instances/${id}/approve`),
  stats: () => api.get('/extraction/stats'),
  diagnosisDistribution: (top = 30) => api.get('/extraction/diagnosis-distribution', { params: { top } }),
  defaultPrompts: () => api.get('/extraction/prompts/defaults'),
}

export const datasets = {
  list: () => api.get('/datasets'),
  create: (data) => api.post('/datasets', data),
  get: (id) => api.get(`/datasets/${id}`),
  items: (id, params) => api.get(`/datasets/${id}/items`, { params }),
  exportUrl: (id) => `/api/datasets/${id}/export`,
  delete: (id) => api.delete(`/datasets/${id}`),
  split: (id, data) => api.post(`/datasets/${id}/split`, data),
}

export const training = {
  listExperiments: () => api.get('/training/experiments'),
  createExperiment: (data) => api.post('/training/experiments', data),
  getExperiment: (id) => api.get(`/training/experiments/${id}`),
  deleteExperiment: (id) => api.delete(`/training/experiments/${id}`),
  updateStatus: (id, status) => api.patch(`/training/experiments/${id}/status`, null, { params: { status } }),
  startTraining: (id) => api.post(`/training/experiments/${id}/start`),
  stopTraining: (id) => api.post(`/training/experiments/${id}/stop`),
  submitMetrics: (id, data) => api.post(`/training/experiments/${id}/metrics`, data),
  getMetrics: (id) => api.get(`/training/experiments/${id}/metrics`),
  getLogs: (id, params) => api.get(`/training/experiments/${id}/logs`, { params }),
  logsStreamUrl: (id, sinceId = 0) => `/api/training/experiments/${id}/logs/stream?since_id=${sinceId}`,
  gpuInfo: () => api.get('/training/gpu-info'),
  stats: () => api.get('/training/stats'),
}

export const assistant = {
  diagnose: (data) => api.post('/assistant/diagnose', data),
  similarCases: (query) => api.get('/assistant/similar-cases', { params: { query } }),
}

export const assistants = {
  list: () => api.get('/assistants'),
  create: (data) => api.post('/assistants', data),
  get: (id) => api.get(`/assistants/${id}`),
  update: (id, data) => api.patch(`/assistants/${id}`, data),
  delete: (id) => api.delete(`/assistants/${id}`),
  start: (id) => api.post(`/assistants/${id}/start`),
  stop: (id) => api.post(`/assistants/${id}/stop`),
  log: (id, tail = 200) => api.get(`/assistants/${id}/log`, { params: { tail } }),
  fromExperiment: (expId, data) => api.post(`/assistants/from-experiment/${expId}`, data),
  gpuInfo: () => api.get('/assistants/gpu-info'),
}

export const evaluations = {
  list: () => api.get('/evaluations'),
  create: (data) => api.post('/evaluations', data),
  get: (id) => api.get(`/evaluations/${id}`),
  start: (id, mode = 'resume') => api.post(`/evaluations/${id}/start`, null, { params: { restart_mode: mode } }),
  cancel: (id) => api.post(`/evaluations/${id}/cancel`),
  delete: (id) => api.delete(`/evaluations/${id}`),
  items: (id, params) => api.get(`/evaluations/${id}/items`, { params }),
}

export const pretraining = {
  listExperiments: () => api.get('/pretraining/experiments'),
  createExperiment: (data) => api.post('/pretraining/experiments', data),
  getExperiment: (id) => api.get(`/pretraining/experiments/${id}`),
  deleteExperiment: (id) => api.delete(`/pretraining/experiments/${id}`),
  start: (id) => api.post(`/pretraining/experiments/${id}/start`),
  stop: (id) => api.post(`/pretraining/experiments/${id}/stop`),
  getMetrics: (id) => api.get(`/pretraining/experiments/${id}/metrics`),
  getLogs: (id, params) => api.get(`/pretraining/experiments/${id}/logs`, { params }),
  logsStreamUrl: (id, sinceId = 0) => `/api/pretraining/experiments/${id}/logs/stream?since_id=${sinceId}`,
  previewCorpus: (filt) => api.post('/pretraining/preview-corpus', filt),
  stats: () => api.get('/pretraining/stats'),
}

export const globalStats = () => api.get('/stats')

export default api
