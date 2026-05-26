import { useEffect, useRef, useState, useCallback } from 'react'
import { training, datasets, assistants as assistantsApi, evaluations } from '../api/client'
import {
  Plus, RefreshCw, Play, Square, Trash2, ChevronDown, ChevronRight,
  Cpu, BarChart2, Terminal, AlertCircle, CheckCircle2, Loader2, Clock,
  Award, Eye, X,
} from 'lucide-react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer,
} from 'recharts'
import PretrainingTab from './Pretraining'

// ─────────────────────────────── constants ───────────────────────────────────

const STATUS_CFG = {
  pending:   { label: '待启动', cls: 'bg-slate-100 text-slate-600' },
  running:   { label: '训练中', cls: 'bg-blue-100 text-blue-700' },
  completed: { label: '已完成', cls: 'bg-green-100 text-green-700' },
  failed:    { label: '失败',   cls: 'bg-red-100 text-red-700' },
  stopped:   { label: '已停止', cls: 'bg-amber-100 text-amber-700' },
}

// ─────────────────────────────── GPU Info Banner ─────────────────────────────

function GpuBanner() {
  const [gpus, setGpus] = useState(null)

  useEffect(() => {
    training.gpuInfo()
      .then(r => setGpus(r.data.gpus))
      .catch(() => setGpus([]))
  }, [])

  if (gpus === null) return null
  if (gpus.length === 0) return (
    <div className="mb-4 flex items-center gap-2 px-4 py-2.5 bg-amber-50 border border-amber-200 rounded-xl text-sm text-amber-700">
      <AlertCircle size={15} />
      <span>未检测到 NVIDIA GPU，训练将使用 CPU（速度极慢）</span>
    </div>
  )

  return (
    <div className="mb-4 flex flex-wrap gap-3">
      {gpus.map(g => {
        const usedPct = Math.round(g.memory_used_mb / g.memory_total_mb * 100)
        return (
          <div key={g.index} className="flex items-center gap-3 px-4 py-2.5 bg-green-50 border border-green-200 rounded-xl text-xs">
            <Cpu size={14} className="text-green-600 shrink-0" />
            <div>
              <p className="font-medium text-green-800">{g.name}</p>
              <p className="text-green-600 mt-0.5">
                显存 {g.memory_used_mb}MB / {g.memory_total_mb}MB ({usedPct}%) &nbsp;|&nbsp;
                GPU利用率 {g.utilization_pct}% &nbsp;|&nbsp; {g.temperature_c}°C
              </p>
            </div>
          </div>
        )
      })}
    </div>
  )
}

// ─────────────────────────────── Create Modal ────────────────────────────────

const DEFAULT_CFG = {
  learning_rate: 2e-4,
  num_epochs: 3,
  batch_size: 4,
  gradient_accumulation_steps: 4,
  max_seq_length: 2048,
  warmup_ratio: 0.05,
  weight_decay: 0.01,
  logging_steps: 10,
  eval_steps: 50,
  save_steps: 100,
  use_lora: true,
  lora_r: 16,
  lora_alpha: 32,
  lora_dropout: 0.05,
  lora_target_modules: 'all-linear',
  use_4bit: false,
  use_bf16: false,
}

function CreateExpModal({ onClose, onCreated }) {
  const [form, setForm] = useState({
    name: '',
    base_model: '',
    output_dir: '',
    dataset_id: '',
    train_ratio: 90,
  })
  const [cfg, setCfg] = useState(DEFAULT_CFG)
  const [dsList, setDsList] = useState([])
  const [loading, setLoading] = useState(false)
  const [tab, setTab] = useState('basic')   // basic | hparam | lora | hardware

  // ── Hardware (GPU) selection ─────────────────────────────────────────────
  // gpuMode: 'auto' | 'single' | 'multi' | 'cpu'
  //   auto   → gpu_ids = null  (let CUDA_VISIBLE_DEVICES from server env take over)
  //   cpu    → gpu_ids = []
  //   single → gpu_ids = [pickedSingle]
  //   multi  → gpu_ids = [...pickedMulti]
  const [gpuMode, setGpuMode] = useState('auto')
  const [availableGpus, setAvailableGpus] = useState([])      // [{index,name,memory_*}]
  const [gpuLoading, setGpuLoading] = useState(false)
  const [pickedSingle, setPickedSingle] = useState(null)      // int | null
  const [pickedMulti, setPickedMulti] = useState([])          // int[]
  const [manualIds, setManualIds] = useState('')              // fallback when no GPUs reported

  const refreshGpus = useCallback(() => {
    setGpuLoading(true)
    training.gpuInfo()
      .then(r => setAvailableGpus(r.data.gpus || []))
      .catch(() => setAvailableGpus([]))
      .finally(() => setGpuLoading(false))
  }, [])

  useEffect(() => {
    datasets.list().then(r => setDsList(r.data.filter(d => d.status === 'ready')))
    refreshGpus()
  }, [refreshGpus])

  const set = (k, v) => setForm(f => ({ ...f, [k]: v }))
  const setCfgVal = (k, v) => setCfg(c => ({ ...c, [k]: v }))

  // Resolve gpuMode + selections → final gpu_ids value (null | int[])
  const resolveGpuIds = () => {
    if (gpuMode === 'auto') return null
    if (gpuMode === 'cpu')  return []
    // For single/multi, prefer GUI selection; fall back to manual entry.
    if (availableGpus.length > 0) {
      if (gpuMode === 'single') return pickedSingle != null ? [pickedSingle] : null
      return pickedMulti.length ? [...pickedMulti].sort((a, b) => a - b) : null
    }
    const ids = manualIds
      .split(',').map(s => s.trim()).filter(Boolean).map(Number)
      .filter(n => Number.isInteger(n) && n >= 0)
    if (gpuMode === 'single') return ids.length ? [ids[0]] : null
    return ids.length ? ids : null
  }

  const handleSubmit = async () => {
    if (!form.name.trim() || !form.base_model.trim()) return
    setLoading(true)
    try {
      const gpuIds = resolveGpuIds()
      await training.createExperiment({
        name: form.name.trim(),
        base_model: form.base_model.trim(),
        output_dir: form.output_dir.trim() || undefined,
        dataset_id: form.dataset_id ? Number(form.dataset_id) : undefined,
        train_ratio: form.train_ratio / 100,
        config: { ...cfg, gpu_ids: gpuIds },
      })
      onCreated()
      onClose()
    } catch {}
    setLoading(false)
  }

  const inputCls = 'w-full px-3 py-2 text-sm border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500'
  const labelCls = 'text-xs text-slate-500 mb-1 block'
  const numInput = (key, label, min, max, step = 1) => (
    <div>
      <label className={labelCls}>{label}</label>
      <input type="number" min={min} max={max} step={step} className={inputCls}
        value={cfg[key]} onChange={e => setCfgVal(key, step < 1 ? parseFloat(e.target.value) : parseInt(e.target.value))} />
    </div>
  )

  return (
    <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-2xl max-h-[90vh] flex flex-col">
        <div className="px-6 pt-6 pb-4 border-b border-slate-100">
          <h2 className="text-base font-semibold text-slate-800">新建训练实验</h2>
        </div>

        {/* Tabs */}
        <div className="flex gap-1 px-6 pt-3 flex-wrap">
          {[['basic','基本配置'],['hparam','超参数'],['lora','LoRA / 量化'],['hardware','硬件资源']].map(([k, label]) => (
            <button key={k} onClick={() => setTab(k)}
              className={`px-3 py-1.5 text-xs rounded-lg font-medium transition-colors
                ${tab === k ? 'bg-blue-100 text-blue-700' : 'text-slate-500 hover:bg-slate-100'}`}>
              {label}
            </button>
          ))}
        </div>

        <div className="overflow-y-auto flex-1 px-6 py-4">
          {tab === 'basic' && (
            <div className="space-y-3">
              <div>
                <label className={labelCls}>实验名称 <span className="text-red-400">*</span></label>
                <input className={inputCls} value={form.name} onChange={e => set('name', e.target.value)}
                  placeholder="例：Qwen2.5-7B 医学 SFT v1" />
              </div>
              <div>
                <label className={labelCls}>基座模型路径 <span className="text-red-400">*</span></label>
                <input className={inputCls} value={form.base_model} onChange={e => set('base_model', e.target.value)}
                  placeholder="/data/models/Qwen2.5-7B-Instruct 或 HF model id" />
              </div>
              <div>
                <label className={labelCls}>输出路径（留空则自动生成）</label>
                <input className={inputCls} value={form.output_dir} onChange={e => set('output_dir', e.target.value)}
                  placeholder="/data/finetuned/my-model-v1" />
              </div>
              <div>
                <label className={labelCls}>训练数据集</label>
                <select className={inputCls} value={form.dataset_id} onChange={e => set('dataset_id', e.target.value)}>
                  <option value="">请选择数据集</option>
                  {dsList.map(d => (
                    <option key={d.id} value={d.id}>{d.name}（{d.item_count} 条）</option>
                  ))}
                </select>
                {dsList.length === 0 && (
                  <p className="text-xs text-amber-600 mt-1">暂无就绪数据集，请先构建并确认数据集</p>
                )}
              </div>
              <div>
                <label className={labelCls}>
                  训练集 / 验证集比例：{form.train_ratio}% / {100 - form.train_ratio}%
                </label>
                <input type="range" min={70} max={99} step={1} className="w-full accent-blue-600"
                  value={form.train_ratio} onChange={e => set('train_ratio', parseInt(e.target.value))} />
              </div>
            </div>
          )}

          {tab === 'hparam' && (
            <div className="grid grid-cols-2 gap-3">
              {numInput('learning_rate', '学习率', 1e-6, 1e-2, 1e-6)}
              {numInput('num_epochs', '训练轮次 (Epoch)', 1, 50)}
              {numInput('batch_size', '每卡批大小', 1, 64)}
              {numInput('gradient_accumulation_steps', '梯度累积步数', 1, 64)}
              {numInput('max_seq_length', '最大序列长度', 128, 32768)}
              {numInput('logging_steps', '日志记录步数', 1, 500)}
              {numInput('eval_steps', '验证步数间隔', 10, 1000)}
              {numInput('save_steps', '保存步数间隔', 10, 1000)}
              {numInput('warmup_ratio', '预热比例', 0, 0.5, 0.01)}
              {numInput('weight_decay', '权重衰减', 0, 0.1, 0.001)}
            </div>
          )}

          {tab === 'lora' && (
            <div className="space-y-3">
              <div className="flex items-center gap-2">
                <input type="checkbox" id="use_lora" checked={cfg.use_lora}
                  onChange={e => setCfgVal('use_lora', e.target.checked)}
                  className="w-4 h-4 accent-blue-600" />
                <label htmlFor="use_lora" className="text-sm font-medium text-slate-700">启用 LoRA 微调（推荐）</label>
              </div>
              {cfg.use_lora && (
                <div className="grid grid-cols-2 gap-3 pl-6">
                  {numInput('lora_r', 'LoRA Rank (r)', 4, 256)}
                  {numInput('lora_alpha', 'LoRA Alpha', 8, 512)}
                  {numInput('lora_dropout', 'LoRA Dropout', 0, 0.5, 0.01)}
                  <div>
                    <label className={labelCls}>目标模块</label>
                    <input className={inputCls} value={cfg.lora_target_modules}
                      onChange={e => setCfgVal('lora_target_modules', e.target.value)}
                      placeholder="all-linear 或逗号分隔: q_proj,v_proj" />
                  </div>
                </div>
              )}
              <div className="border-t border-slate-100 pt-3 space-y-2">
                <p className="text-xs font-medium text-slate-600">精度设置</p>
                <label className="flex items-center gap-2 text-sm">
                  <input type="checkbox" checked={cfg.use_4bit}
                    onChange={e => setCfgVal('use_4bit', e.target.checked)}
                    className="w-4 h-4 accent-blue-600" />
                  4-bit 量化加载（QLoRA，需要 bitsandbytes）
                </label>
                <label className="flex items-center gap-2 text-sm">
                  <input type="checkbox" checked={cfg.use_bf16}
                    onChange={e => setCfgVal('use_bf16', e.target.checked)}
                    className="w-4 h-4 accent-blue-600" />
                  使用 BF16（Ampere+ 架构推荐，否则用 FP16）
                </label>
              </div>
            </div>
          )}

          {tab === 'hardware' && (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <p className="text-xs text-slate-500">
                  选择训练所使用的计算资源。多卡将自动通过 <code className="px-1 py-0.5 bg-slate-100 rounded">torchrun</code> 启动 DDP。
                </p>
                <button type="button" onClick={refreshGpus}
                  className="flex items-center gap-1 text-xs text-slate-500 hover:text-blue-600">
                  <RefreshCw size={12} className={gpuLoading ? 'animate-spin' : ''} /> 刷新
                </button>
              </div>

              {/* Mode radio */}
              <div className="grid grid-cols-2 gap-2">
                {[
                  ['auto',   '自动',     '不修改 CUDA_VISIBLE_DEVICES，由服务器环境决定'],
                  ['single', '单卡训练', '指定一张 GPU，python 直接启动'],
                  ['multi',  '多卡训练', '选择多张 GPU，使用 torchrun + DDP'],
                  ['cpu',    '仅 CPU',  '设置 CUDA_VISIBLE_DEVICES="" 强制走 CPU（仅调试）'],
                ].map(([k, label, desc]) => (
                  <label key={k}
                    className={`flex items-start gap-2 p-3 rounded-xl border cursor-pointer transition-colors
                      ${gpuMode === k ? 'border-blue-400 bg-blue-50' : 'border-slate-200 hover:bg-slate-50'}`}>
                    <input type="radio" name="gpuMode" className="mt-0.5 accent-blue-600"
                      checked={gpuMode === k} onChange={() => setGpuMode(k)} />
                    <div className="min-w-0">
                      <p className="text-sm font-medium text-slate-700">{label}</p>
                      <p className="text-xs text-slate-400 mt-0.5">{desc}</p>
                    </div>
                  </label>
                ))}
              </div>

              {/* GPU picker */}
              {(gpuMode === 'single' || gpuMode === 'multi') && (
                <div className="border-t border-slate-100 pt-3">
                  {availableGpus.length > 0 ? (
                    <div className="space-y-1.5">
                      <p className="text-xs font-medium text-slate-600 mb-1">
                        {gpuMode === 'single' ? '选择一张 GPU' : '选择参与训练的 GPU（≥2 张）'}
                      </p>
                      {availableGpus.map(g => {
                        const isPicked = gpuMode === 'single'
                          ? pickedSingle === g.index
                          : pickedMulti.includes(g.index)
                        const usedPct = Math.round(g.memory_used_mb / g.memory_total_mb * 100)
                        return (
                          <label key={g.index}
                            className={`flex items-center gap-3 p-2.5 rounded-lg border cursor-pointer transition-colors
                              ${isPicked ? 'border-blue-400 bg-blue-50' : 'border-slate-200 hover:bg-slate-50'}`}>
                            <input
                              type={gpuMode === 'single' ? 'radio' : 'checkbox'}
                              name="gpuPick"
                              checked={isPicked}
                              onChange={() => {
                                if (gpuMode === 'single') {
                                  setPickedSingle(g.index)
                                } else {
                                  setPickedMulti(p =>
                                    p.includes(g.index) ? p.filter(x => x !== g.index) : [...p, g.index])
                                }
                              }}
                              className="accent-blue-600"
                            />
                            <Cpu size={14} className="text-slate-400 shrink-0" />
                            <div className="flex-1 min-w-0">
                              <p className="text-sm text-slate-700 truncate">
                                <span className="font-medium">GPU {g.index}</span>
                                <span className="ml-2 text-slate-500">{g.name}</span>
                              </p>
                              <p className="text-xs text-slate-400 mt-0.5">
                                显存 {g.memory_used_mb}/{g.memory_total_mb} MB ({usedPct}%) ·
                                空闲 {g.memory_free_mb} MB · 利用率 {g.utilization_pct}% · {g.temperature_c}°C
                              </p>
                            </div>
                          </label>
                        )
                      })}
                      {gpuMode === 'multi' && pickedMulti.length === 1 && (
                        <p className="text-xs text-amber-600">至少选择 2 张 GPU 才会启用 DDP；当前只选 1 张将退化为单卡。</p>
                      )}
                    </div>
                  ) : (
                    <div className="space-y-2">
                      <p className="text-xs text-amber-600">
                        当前无法读取 GPU 列表（可能在开发机上）。可手动填写 GPU 索引，提交后由服务器解析。
                      </p>
                      <input type="text"
                        value={manualIds}
                        onChange={e => setManualIds(e.target.value)}
                        placeholder={gpuMode === 'single' ? '例：0' : '例：0,1,2,3'}
                        className="w-full px-3 py-2 text-sm border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500" />
                    </div>
                  )}
                </div>
              )}

              {/* Live preview of resolved gpu_ids */}
              <div className="border-t border-slate-100 pt-3">
                <p className="text-xs text-slate-400">
                  最终下发：
                  <code className="ml-1 px-2 py-0.5 bg-slate-100 rounded text-slate-700">
                    gpu_ids = {(() => {
                      const v = resolveGpuIds()
                      if (v === null) return 'null （auto）'
                      if (v.length === 0) return '[] （CPU）'
                      return `[${v.join(', ')}]`
                    })()}
                  </code>
                </p>
              </div>
            </div>
          )}
        </div>

        <div className="px-6 py-4 border-t border-slate-100 flex justify-end gap-2">
          <button onClick={onClose}
            className="px-4 py-2 text-sm border border-slate-200 rounded-lg hover:bg-slate-50">取消</button>
          <button onClick={handleSubmit} disabled={loading || !form.name.trim() || !form.base_model.trim()}
            className="px-4 py-2 text-sm bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50">
            {loading ? '创建中...' : '创建实验'}
          </button>
        </div>
      </div>
    </div>
  )
}

// ─────────────────────────────── Metrics Charts ──────────────────────────────

function MetricsPanel({ expId }) {
  const [metrics, setMetrics] = useState([])
  const [activeChart, setActiveChart] = useState('loss')

  useEffect(() => {
    const poll = () => training.getMetrics(expId).then(r => setMetrics(r.data)).catch(() => {})
    poll()
    const t = setInterval(poll, 4000)
    return () => clearInterval(t)
  }, [expId])

  if (metrics.length === 0) {
    return <p className="text-xs text-slate-400 py-6 text-center">暂无训练指标，启动训练后自动更新</p>
  }

  const hasEval = metrics.some(m => m.eval_loss != null)
  const hasLr = metrics.some(m => m.learning_rate != null)
  const hasGrad = metrics.some(m => m.extra_metrics?.grad_norm != null)

  const chartData = metrics.map(m => ({
    step: m.step,
    epoch: m.epoch ? m.epoch.toFixed(2) : null,
    train_loss: m.train_loss,
    eval_loss: m.eval_loss,
    learning_rate: m.learning_rate,
    grad_norm: m.extra_metrics?.grad_norm,
  }))

  const tabs = [
    { key: 'loss', label: 'Loss 曲线' },
    ...(hasLr ? [{ key: 'lr', label: '学习率' }] : []),
    ...(hasGrad ? [{ key: 'grad', label: 'Grad Norm' }] : []),
  ]

  return (
    <div className="mt-3">
      <div className="flex gap-1 mb-2">
        {tabs.map(t => (
          <button key={t.key} onClick={() => setActiveChart(t.key)}
            className={`px-2.5 py-1 text-xs rounded-md font-medium transition-colors
              ${activeChart === t.key ? 'bg-blue-100 text-blue-700' : 'text-slate-400 hover:bg-slate-100'}`}>
            {t.label}
          </button>
        ))}
        <span className="ml-auto text-xs text-slate-400 self-center">{metrics.length} 个数据点</span>
      </div>

      <div className="h-48">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={{ top: 4, right: 16, left: -10, bottom: 4 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
            <XAxis dataKey="step" tick={{ fontSize: 10 }} label={{ value: 'step', position: 'insideBottomRight', offset: 0, fontSize: 10 }} />
            <YAxis tick={{ fontSize: 10 }} domain={['auto', 'auto']} />
            <Tooltip contentStyle={{ fontSize: 11 }} formatter={(v) => typeof v === 'number' ? v.toFixed(6) : v} />
            <Legend wrapperStyle={{ fontSize: 11 }} />

            {/* NOTE: do NOT wrap multiple <Line> in a Fragment — Recharts
                cannot detect Line components nested inside <>...</>, so the
                line silently doesn't render. Each <Line> must be a direct
                child of <LineChart>. */}
            {activeChart === 'loss' && metrics.some(m => m.train_loss != null) && (
              <Line type="monotone" dataKey="train_loss" stroke="#3b82f6" dot={false}
                name="训练损失" strokeWidth={1.5} connectNulls isAnimationActive={false} />
            )}
            {activeChart === 'loss' && hasEval && (
              <Line type="monotone" dataKey="eval_loss" stroke="#10b981" dot={false}
                name="验证损失" strokeWidth={1.5} connectNulls isAnimationActive={false} />
            )}
            {activeChart === 'lr' && (
              <Line type="monotone" dataKey="learning_rate" stroke="#f59e0b" dot={false}
                name="学习率" strokeWidth={1.5} connectNulls isAnimationActive={false} />
            )}
            {activeChart === 'grad' && (
              <Line type="monotone" dataKey="grad_norm" stroke="#8b5cf6" dot={false}
                name="Grad Norm" strokeWidth={1.5} connectNulls isAnimationActive={false} />
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

// ─────────────────────────────── Log Console ─────────────────────────────────

const LOG_LEVEL_CLS = {
  info:    'text-slate-300',
  metrics: 'text-green-400',
  warning: 'text-amber-400',
  error:   'text-red-400',
}

function LogConsole({ expId, isRunning }) {
  const [logs, setLogs] = useState([])
  const [connected, setConnected] = useState(false)
  const bottomRef = useRef(null)
  const esRef = useRef(null)
  const logsRef = useRef([])
  const lastIdRef = useRef(0)

  const scrollToBottom = useCallback(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [])

  useEffect(() => {
    // Load existing logs first
    training.getLogs(expId, { limit: 300 }).then(r => {
      logsRef.current = r.data
      setLogs([...r.data])
      if (r.data.length > 0) lastIdRef.current = r.data[r.data.length - 1].id
      setTimeout(scrollToBottom, 100)
    }).catch(() => {})
  }, [expId])

  useEffect(() => {
    if (!isRunning) {
      setConnected(false)
      return
    }

    const connectSSE = () => {
      const url = training.logsStreamUrl(expId, lastIdRef.current)
      const es = new EventSource(url)
      esRef.current = es
      setConnected(true)

      es.onmessage = (e) => {
        try {
          const data = JSON.parse(e.data)
          if (data.error || data.done) return
          logsRef.current = [...logsRef.current, data]
          setLogs([...logsRef.current])
          lastIdRef.current = data.id || lastIdRef.current
          scrollToBottom()
        } catch {}
      }

      es.addEventListener('done', (e) => {
        setConnected(false)
        es.close()
      })

      es.onerror = () => {
        setConnected(false)
        es.close()
        // Retry after 3 seconds
        setTimeout(connectSSE, 3000)
      }
    }

    connectSSE()
    return () => { esRef.current?.close() }
  }, [expId, isRunning])

  const formatMsg = (msg) => {
    // Try to parse JSON for pretty display
    try {
      const d = JSON.parse(msg)
      if (d.type === 'metrics') {
        const parts = []
        if (d.step != null)       parts.push(`step=${d.step}`)
        if (d.epoch != null)      parts.push(`epoch=${Number(d.epoch).toFixed(2)}`)
        if (d.train_loss != null) parts.push(`train_loss=${Number(d.train_loss).toFixed(4)}`)
        if (d.eval_loss != null)  parts.push(`eval_loss=${Number(d.eval_loss).toFixed(4)}`)
        if (d.learning_rate)      parts.push(`lr=${Number(d.learning_rate).toExponential(3)}`)
        if (d.grad_norm != null)  parts.push(`grad_norm=${Number(d.grad_norm).toFixed(4)}`)
        return `[metrics] ${parts.join('  ')}`
      }
      if (d.type === 'progress') return `[progress] ${d.message}`
      if (d.type === 'error')    return `[error] ${d.message}`
      if (d.type === 'info')     return `[info] ${JSON.stringify(d)}`
      if (d.type === 'completed') return `[completed] 模型已保存: ${d.output_dir}`
      return msg
    } catch {
      return msg
    }
  }

  return (
    <div className="mt-3">
      <div className="flex items-center gap-2 mb-1.5">
        <Terminal size={13} className="text-slate-400" />
        <span className="text-xs font-medium text-slate-500">训练日志</span>
        {isRunning && (
          <span className={`flex items-center gap-1 text-xs ml-auto ${connected ? 'text-green-500' : 'text-amber-500'}`}>
            <span className={`w-1.5 h-1.5 rounded-full ${connected ? 'bg-green-500 animate-pulse' : 'bg-amber-400'}`} />
            {connected ? '实时连接' : '重连中...'}
          </span>
        )}
      </div>
      <div className="bg-slate-900 rounded-xl p-3 h-52 overflow-y-auto font-mono text-xs leading-relaxed">
        {logs.length === 0 && (
          <span className="text-slate-500">等待训练输出...</span>
        )}
        {logs.map((lg, i) => (
          <div key={lg.id ?? i} className={`${LOG_LEVEL_CLS[lg.level] ?? 'text-slate-300'} whitespace-pre-wrap break-all`}>
            {formatMsg(lg.message)}
          </div>
        ))}
        <div ref={bottomRef} />
      </div>
    </div>
  )
}

// ─────────────────────────────── Experiment Card ─────────────────────────────

function ProgressBar({ current, total }) {
  if (!total || total === 0) return null
  const pct = Math.min(100, Math.round((current / total) * 100))
  return (
    <div className="mt-2">
      <div className="flex justify-between text-xs text-slate-400 mb-1">
        <span>步骤 {current ?? 0} / {total}</span>
        <span>{pct}%</span>
      </div>
      <div className="h-1.5 bg-slate-100 rounded-full overflow-hidden">
        <div className="h-full bg-blue-500 rounded-full transition-all duration-500"
          style={{ width: `${pct}%` }} />
      </div>
    </div>
  )
}

function ExpCard({ exp, onRefresh }) {
  const [expanded, setExpanded] = useState(false)
  const [activeTab, setActiveTab] = useState('metrics')
  const [acting, setActing] = useState(false)
  const [expData, setExpData] = useState(exp)
  const pollRef = useRef(null)

  // Poll for status updates while running
  useEffect(() => {
    if (expData.status === 'running') {
      pollRef.current = setInterval(() => {
        training.getExperiment(expData.id).then(r => setExpData(r.data)).catch(() => {})
      }, 3000)
    }
    return () => clearInterval(pollRef.current)
  }, [expData.status, expData.id])

  const handleStart = async () => {
    setActing(true)
    try {
      await training.startTraining(expData.id)
      await training.getExperiment(expData.id).then(r => setExpData(r.data))
    } catch (e) {
      alert(e?.response?.data?.detail || '启动失败')
    }
    setActing(false)
  }

  const handleStop = async () => {
    if (!confirm('确定要停止训练吗？')) return
    setActing(true)
    try {
      await training.stopTraining(expData.id)
      await training.getExperiment(expData.id).then(r => setExpData(r.data))
    } catch {}
    setActing(false)
  }

  const handleDelete = async () => {
    if (!confirm(`确定删除实验「${expData.name}」？此操作不可撤销。`)) return
    setActing(true)
    try {
      await training.deleteExperiment(expData.id)
      onRefresh()
    } catch (e) {
      alert(e?.response?.data?.detail || '删除失败')
      setActing(false)
    }
  }

  const s = STATUS_CFG[expData.status] || STATUS_CFG.pending
  const cfg = expData.config || {}
  const isRunning = expData.status === 'running'

  // gpu_ids semantics: undefined/null=auto, []=cpu, [n]=single, [n,m,..]=multi
  const gpuLabel = (() => {
    const ids = cfg.gpu_ids
    if (ids === undefined || ids === null) return '自动'
    if (Array.isArray(ids) && ids.length === 0) return '仅 CPU'
    if (Array.isArray(ids) && ids.length === 1) return `单卡 GPU ${ids[0]}`
    if (Array.isArray(ids))                     return `多卡 DDP GPU [${ids.join(',')}]`
    return String(ids)
  })()

  // baseline vs final accuracy comparison
  const baseAcc = expData.baseline_eval?.token_accuracy
  const finalAcc = expData.final_eval?.token_accuracy
  const accDelta = (baseAcc != null && finalAcc != null) ? (finalAcc - baseAcc) : null

  return (
    <div className="bg-white rounded-2xl shadow-sm border border-slate-100 overflow-hidden">
      {/* Header */}
      <div className="px-5 py-4">
        <div className="flex items-start justify-between gap-3">
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 flex-wrap">
              <span className="font-semibold text-slate-800 truncate">{expData.name}</span>
              <span className={`px-2 py-0.5 rounded-full text-xs font-medium shrink-0 ${s.cls}`}>
                {isRunning && <span className="inline-block w-1.5 h-1.5 rounded-full bg-blue-500 animate-pulse mr-1 align-middle" />}
                {s.label}
              </span>
            </div>
            <div className="mt-1 text-xs text-slate-400 flex flex-wrap gap-x-4 gap-y-0.5">
              <span>基座：{expData.base_model}</span>
              {expData.dataset_name && (
                <span>数据集：<span className="text-slate-600 font-medium">{expData.dataset_name}</span></span>
              )}
              {expData.best_eval_loss != null && (
                <span className="text-green-600 font-medium">最优验证损失：{expData.best_eval_loss.toFixed(4)}</span>
              )}
              {(baseAcc != null || finalAcc != null) && (
                <span className="text-emerald-600 font-medium">
                  验证集准确率：
                  <span className="text-slate-500 font-normal">基座 </span>
                  {baseAcc != null ? `${(baseAcc * 100).toFixed(2)}%` : 'N/A'}
                  <span className="text-slate-400 mx-1">→</span>
                  <span className="text-slate-500 font-normal">微调 </span>
                  {finalAcc != null ? `${(finalAcc * 100).toFixed(2)}%` : 'N/A'}
                  {accDelta != null && (
                    <span className={`ml-1.5 px-1.5 py-0.5 rounded text-xs ${accDelta >= 0 ? 'bg-emerald-100 text-emerald-700' : 'bg-red-100 text-red-700'}`}>
                      {accDelta >= 0 ? '+' : ''}{(accDelta * 100).toFixed(2)} pp
                    </span>
                  )}
                </span>
              )}
              {cfg.num_epochs && <span>Epoch×{cfg.num_epochs}</span>}
              {cfg.learning_rate && <span>lr={cfg.learning_rate}</span>}
              {cfg.use_lora && <span className="text-purple-600">LoRA r={cfg.lora_r}</span>}
              {cfg.use_4bit && <span className="text-orange-500">4-bit</span>}
              <span className="text-blue-600">{gpuLabel}</span>
              <span>创建：{expData.created_at?.slice(0, 10)}</span>
            </div>

            {isRunning && (
              <ProgressBar current={expData.current_step} total={expData.total_steps} />
            )}
            {expData.error_message && (
              <p className="mt-1 text-xs text-red-500 flex items-center gap-1">
                <AlertCircle size={11} /> {expData.error_message}
              </p>
            )}
          </div>

          {/* Actions */}
          <div className="flex items-center gap-1.5 shrink-0">
            {(expData.status === 'pending' || expData.status === 'stopped' || expData.status === 'failed') && (
              <button onClick={handleStart} disabled={acting}
                className="flex items-center gap-1 px-3 py-1.5 text-xs bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50">
                {acting ? <Loader2 size={12} className="animate-spin" /> : <Play size={12} />}
                {expData.status === 'pending' ? '启动训练' : '重新训练'}
              </button>
            )}
            {isRunning && (
              <button onClick={handleStop} disabled={acting}
                className="flex items-center gap-1 px-3 py-1.5 text-xs bg-red-500 text-white rounded-lg hover:bg-red-600 disabled:opacity-50">
                {acting ? <Loader2 size={12} className="animate-spin" /> : <Square size={12} />}
                停止
              </button>
            )}
            <button onClick={() => setExpanded(!expanded)}
              className="flex items-center gap-1 px-3 py-1.5 text-xs border border-slate-200 rounded-lg hover:bg-slate-50 text-slate-600">
              <BarChart2 size={12} />
              {expanded ? '收起' : '详情'}
              {expanded ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
            </button>
            {!isRunning && (
              <button onClick={handleDelete} disabled={acting}
                className="p-1.5 text-slate-400 hover:text-red-500 hover:bg-red-50 rounded-lg transition-colors">
                <Trash2 size={14} />
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Expanded Panel */}
      {expanded && (
        <div className="border-t border-slate-100 px-5 pb-5">
          <div className="flex gap-1 pt-3 mb-2">
            {[['metrics', '指标曲线'], ['logs', '训练日志'], ['config', '实验配置']].map(([k, label]) => (
              <button key={k} onClick={() => setActiveTab(k)}
                className={`flex items-center gap-1.5 px-3 py-1.5 text-xs rounded-lg font-medium transition-colors
                  ${activeTab === k ? 'bg-blue-100 text-blue-700' : 'text-slate-400 hover:bg-slate-100'}`}>
                {k === 'metrics' && <BarChart2 size={12} />}
                {k === 'logs' && <Terminal size={12} />}
                {label}
              </button>
            ))}
          </div>

          {activeTab === 'metrics' && <MetricsPanel expId={expData.id} />}
          {activeTab === 'logs' && <LogConsole expId={expData.id} isRunning={isRunning} />}
          {activeTab === 'config' && (
            <div className="mt-2 bg-slate-50 rounded-xl p-4">
              <dl className="grid grid-cols-2 gap-x-6 gap-y-1.5 text-xs">
                {[
                  ['基座模型', expData.base_model],
                  ['训练数据集', expData.dataset_name || (expData.dataset_id ? `#${expData.dataset_id}` : '-')],
                  ['权重保存目录', expData.final_output_dir || cfg.output_dir || '自动生成'],
                  ['训练集比例', `${Math.round((cfg.train_ratio ?? 0.9) * 100)}%`],
                  ['学习率', cfg.learning_rate],
                  ['Epoch', cfg.num_epochs],
                  ['批大小', cfg.batch_size],
                  ['梯度累积', cfg.gradient_accumulation_steps],
                  ['最大序列长度', cfg.max_seq_length],
                  ['LoRA', cfg.use_lora ? `r=${cfg.lora_r} α=${cfg.lora_alpha}` : '关闭'],
                  ['4-bit量化', cfg.use_4bit ? '是' : '否'],
                  ['精度', cfg.use_bf16 ? 'BF16' : 'FP16'],
                  ['GPU 资源', gpuLabel],
                  ['Log步数', cfg.logging_steps],
                  ['最优验证损失',
                    expData.best_eval_loss != null ? expData.best_eval_loss.toFixed(6) : '-'],
                  ['基座 token 准确率',
                    expData.baseline_eval?.token_accuracy != null
                      ? `${(expData.baseline_eval.token_accuracy * 100).toFixed(2)}% （${expData.baseline_eval.eval_samples ?? '?'} 样本 / ${expData.baseline_eval.eval_tokens ?? '?'} tokens）`
                      : (expData.baseline_eval
                          ? '评估异常，请查看日志'
                          : (expData.status === 'completed' ? '未生成（请检查训练集划分或日志）' : '训练开始时生成'))],
                  ['微调后 token 准确率',
                    expData.final_eval?.token_accuracy != null
                      ? `${(expData.final_eval.token_accuracy * 100).toFixed(2)}% （${expData.final_eval.eval_samples ?? '?'} 样本 / ${expData.final_eval.eval_tokens ?? '?'} tokens）`
                      : (expData.final_eval
                          ? '评估异常，请查看日志'
                          : (expData.status === 'completed' ? '未生成（请检查训练集划分或日志）' : '训练完成后生成'))],
                  ['准确率提升',
                    accDelta != null
                      ? `${accDelta >= 0 ? '+' : ''}${(accDelta * 100).toFixed(2)} pp`
                      : '-'],
                ].map(([k, v]) => (
                  <div key={k} className="flex gap-2">
                    <dt className="text-slate-400 shrink-0">{k}：</dt>
                    <dd className="text-slate-700 font-medium break-all">{String(v ?? '-')}</dd>
                  </div>
                ))}
              </dl>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// ─────────────────────────────── Evaluation Panel ───────────────────────────────

const EVAL_STATUS = {
  pending:   { label: '待运行', cls: 'bg-slate-100 text-slate-600' },
  running:   { label: '评估中', cls: 'bg-blue-100 text-blue-700' },
  completed: { label: '已完成', cls: 'bg-green-100 text-green-700' },
  failed:    { label: '失败',   cls: 'bg-red-100 text-red-700' },
  cancelled: { label: '已取消', cls: 'bg-amber-100 text-amber-700' },
}

const PHASE_LABEL = {
  pending:               '阶段 0/2 待启动',
  generating:            '阶段 1/2 候选/基线并行生成回答',
  judging:               '阶段 2/2 LLM 评分',
  done:                  '已完成',
  // legacy phase names from earlier 3-phase design (display gracefully)
  generating_candidate:  '阶段 1/2 生成 candidate 回答',
  generating_baseline:   '阶段 1/2 生成 baseline 回答',
}

function CreateEvalModal({ datasetList, assistantList, onClose, onCreated }) {
  const [form, setForm] = useState({
    name: '',
    dataset_id: '',
    candidate_assistant_id: '',
    baseline_assistant_id: '',
    judge_assistant_id: '',
    sample_limit: 30,
  })
  const [loading, setLoading] = useState(false)
  const [err, setErr] = useState('')

  const handleSubmit = async () => {
    setErr('')
    if (!form.name.trim()) return
    if (!form.dataset_id) { setErr('请选择测试数据集'); return }
    if (!form.candidate_assistant_id) { setErr('请选择 candidate 助手'); return }
    if (!form.judge_assistant_id) { setErr('请选择 judge 助手'); return }
    setLoading(true)
    try {
      await evaluations.create({
        name: form.name.trim(),
        dataset_id: Number(form.dataset_id),
        candidate_assistant_id: Number(form.candidate_assistant_id),
        baseline_assistant_id: form.baseline_assistant_id ? Number(form.baseline_assistant_id) : undefined,
        judge_assistant_id: Number(form.judge_assistant_id),
        sample_limit: form.sample_limit ? Number(form.sample_limit) : undefined,
        auto_start: true,
      })
      onCreated()
      onClose()
    } catch (e) {
      setErr(e?.response?.data?.detail || '创建失败')
    }
    setLoading(false)
  }

  const inputCls = 'w-full px-3 py-2 text-sm border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500'
  const labelCls = 'text-xs text-slate-500 mb-1 block'

  const opt = (a) => `${a.type === 'local' ? '🖥' : '☁'} ${a.name} (${a.model_name})${a.status !== 'running' ? ' - ' + a.status : ''}`

  return (
    <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-2xl shadow-xl w-full max-w-lg p-6 max-h-[90vh] overflow-y-auto">
        <h2 className="text-base font-semibold mb-1">新建模型评估</h2>
        <p className="text-xs text-slate-500 mb-1">使用 LLM-as-Judge 在指定数据集上对比 candidate 与 baseline 模型</p>
        <div className="mb-4 px-3 py-2 bg-blue-50 border border-blue-100 rounded text-[11px] text-blue-700 leading-relaxed">
          <b>两阶段执行：</b><br />
          ① <b>生成阶段</b>：candidate（必需）+ baseline（如选）<b>同时运行</b>并并行生成回答；judge 此时可以离线<br />
          ② <b>评分阶段</b>：judge 必须运行，candidate / baseline 可以离线（评分用的是缓存的回答）<br />
          阶段切换时可在「助手管理」停掉旧模型、启动新模型，再点卡片上的「续跑」继续，无需重头跑。
        </div>
        <div className="space-y-3">
          <div>
            <label className={labelCls}>评估任务名称 *</label>
            <input className={inputCls} value={form.name} onChange={e => setForm(f => ({ ...f, name: e.target.value }))}
              placeholder="例：QA 测试集 - 微调 vs 基座 v1" />
          </div>
          <div>
            <label className={labelCls}>测试数据集 *</label>
            <select className={inputCls} value={form.dataset_id}
              onChange={e => setForm(f => ({ ...f, dataset_id: e.target.value }))}>
              <option value="">请选择</option>
              {datasetList.filter(d => d.status === 'ready').map(d => (
                <option key={d.id} value={d.id}>{d.name}（{d.item_count} 条）</option>
              ))}
            </select>
          </div>
          <div>
            <label className={labelCls}>Candidate 助手（被测模型，通常是微调模型）*</label>
            <select className={inputCls} value={form.candidate_assistant_id}
              onChange={e => setForm(f => ({ ...f, candidate_assistant_id: e.target.value }))}>
              <option value="">请选择</option>
              {assistantList.map(a => <option key={a.id} value={a.id}>{opt(a)}</option>)}
            </select>
          </div>
          <div>
            <label className={labelCls}>Baseline 助手（对比基线，通常是基座模型，可选）</label>
            <select className={inputCls} value={form.baseline_assistant_id}
              onChange={e => setForm(f => ({ ...f, baseline_assistant_id: e.target.value }))}>
              <option value="">— 不对比 —</option>
              {assistantList.map(a => <option key={a.id} value={a.id}>{opt(a)}</option>)}
            </select>
          </div>
          <div>
            <label className={labelCls}>Judge 助手（充当评分员的 LLM）*</label>
            <select className={inputCls} value={form.judge_assistant_id}
              onChange={e => setForm(f => ({ ...f, judge_assistant_id: e.target.value }))}>
              <option value="">请选择</option>
              {assistantList.map(a => <option key={a.id} value={a.id}>{opt(a)}</option>)}
            </select>
          </div>
          <div>
            <label className={labelCls}>样本数上限（节省调用，0 或留空表示全部）</label>
            <input type="number" min="0" className={inputCls} value={form.sample_limit}
              onChange={e => setForm(f => ({ ...f, sample_limit: e.target.value }))} placeholder="30" />
          </div>
        </div>
        {err && <p className="mt-3 text-xs text-red-500">{err}</p>}
        <div className="flex justify-end gap-2 mt-5">
          <button onClick={onClose}
            className="px-4 py-2 text-sm border border-slate-200 rounded-lg hover:bg-slate-50">取消</button>
          <button onClick={handleSubmit} disabled={loading}
            className="px-4 py-2 text-sm bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50">
            {loading ? '创建中...' : '创建并启动'}
          </button>
        </div>
      </div>
    </div>
  )
}

function EvaluationItemsModal({ run, onClose }) {
  const [data, setData] = useState({ items: [], total: 0 })
  const [page, setPage] = useState(1)
  useEffect(() => {
    evaluations.items(run.id, { page, page_size: 5 }).then(r => setData(r.data)).catch(() => {})
  }, [run.id, page])
  const totalPages = Math.max(1, Math.ceil((data.total || 0) / 5))
  return (
    <div className="fixed inset-0 bg-black/40 z-50 flex items-stretch justify-end" onClick={onClose}>
      <div className="bg-white w-full max-w-3xl h-full shadow-2xl flex flex-col" onClick={e => e.stopPropagation()}>
        <div className="px-5 py-4 border-b border-slate-100 flex items-start gap-3">
          <Award size={18} className="text-amber-600 mt-0.5 shrink-0" />
          <div className="flex-1 min-w-0">
            <p className="font-semibold text-slate-800 truncate">{run.name} — 评估明细</p>
            <p className="text-xs text-slate-400 mt-0.5">
              {run.dataset_name} · candidate: {run.candidate_name} {run.baseline_name && ` · baseline: ${run.baseline_name}`} · judge: {run.judge_name}
            </p>
          </div>
          <button onClick={onClose} className="p-1.5 text-slate-400 hover:text-slate-700 hover:bg-slate-100 rounded-lg">
            <X size={16} />
          </button>
        </div>
        <div className="flex-1 overflow-y-auto px-5 py-4 space-y-3">
          {data.items.map(it => (
            <div key={it.id} className="border border-slate-100 rounded-xl p-3 text-xs space-y-2">
              <div>
                <p className="text-slate-400 mb-0.5">问题</p>
                <p className="text-slate-700 whitespace-pre-wrap">{it.instruction}</p>
              </div>
              <div>
                <p className="text-slate-400 mb-0.5">参考答案</p>
                <p className="text-slate-600 whitespace-pre-wrap line-clamp-4">{it.expected_output}</p>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                <div className="p-2 bg-blue-50 rounded-lg">
                  <p className="text-slate-500 text-[11px] mb-1">
                    Candidate {it.candidate_score != null && (
                      <span className={`ml-1 px-1.5 py-0.5 rounded ${it.candidate_score >= 3 ? 'bg-green-200 text-green-800' : 'bg-red-200 text-red-800'}`}>
                        {it.candidate_score.toFixed(1)} / 5
                      </span>
                    )}
                  </p>
                  <p className="text-slate-700 whitespace-pre-wrap line-clamp-6">{it.candidate_response || '—'}</p>
                  {it.candidate_reasoning && <p className="text-slate-400 text-[11px] mt-1">📝 {it.candidate_reasoning}</p>}
                </div>
                {it.baseline_response != null && (
                  <div className="p-2 bg-amber-50 rounded-lg">
                    <p className="text-slate-500 text-[11px] mb-1">
                      Baseline {it.baseline_score != null && (
                        <span className={`ml-1 px-1.5 py-0.5 rounded ${it.baseline_score >= 3 ? 'bg-green-200 text-green-800' : 'bg-red-200 text-red-800'}`}>
                          {it.baseline_score.toFixed(1)} / 5
                        </span>
                      )}
                    </p>
                    <p className="text-slate-700 whitespace-pre-wrap line-clamp-6">{it.baseline_response || '—'}</p>
                    {it.baseline_reasoning && <p className="text-slate-400 text-[11px] mt-1">📝 {it.baseline_reasoning}</p>}
                  </div>
                )}
              </div>
              {it.error_message && <p className="text-red-500 text-[11px]">⚠ {it.error_message}</p>}
            </div>
          ))}
          {data.items.length === 0 && (
            <p className="text-center text-slate-400 py-12">暂无评估结果</p>
          )}
        </div>
        {totalPages > 1 && (
          <div className="px-5 py-3 border-t border-slate-100 flex items-center justify-between text-xs text-slate-500">
            <span>第 {page} / {totalPages} 页 · 共 {data.total} 条</span>
            <div className="flex gap-2">
              <button disabled={page <= 1} onClick={() => setPage(p => p - 1)}
                className="px-2 py-1 border border-slate-200 rounded hover:bg-slate-50 disabled:opacity-40">上一页</button>
              <button disabled={page >= totalPages} onClick={() => setPage(p => p + 1)}
                className="px-2 py-1 border border-slate-200 rounded hover:bg-slate-50 disabled:opacity-40">下一页</button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

function EvalRunCard({ run, onChanged, onView }) {
  const s = EVAL_STATUS[run.status] || EVAL_STATUS.pending
  const phaseLabel = PHASE_LABEL[run.phase] || PHASE_LABEL.pending
  const handleCancel = async () => {
    if (!confirm('取消该评估？')) return
    try { await evaluations.cancel(run.id); onChanged() } catch (e) { alert(e?.response?.data?.detail || '取消失败') }
  }
  const handleDelete = async () => {
    if (!confirm(`删除评估「${run.name}」？`)) return
    try { await evaluations.delete(run.id); onChanged() } catch (e) { alert(e?.response?.data?.detail || '删除失败') }
  }
  const handleResume = async () => {
    try { await evaluations.start(run.id, 'resume'); onChanged() } catch (e) { alert(e?.response?.data?.detail || '续跑失败') }
  }
  const handleFreshRestart = async () => {
    if (!confirm('完全重跑会丢弃已生成的所有回答和评分，确定吗？')) return
    try { await evaluations.start(run.id, 'fresh'); onChanged() } catch (e) { alert(e?.response?.data?.detail || '重跑失败') }
  }

  const pct = run.progress_total > 0 ? Math.round(run.progress_done / run.progress_total * 100) : 0
  const cScore = run.candidate_score
  const bScore = run.baseline_score
  const delta = (cScore != null && bScore != null) ? (cScore - bScore) : null

  return (
    <div className="bg-white rounded-xl shadow-sm border border-slate-100 p-4">
      <div className="flex items-start justify-between gap-3">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="font-semibold text-slate-800 truncate">{run.name}</span>
            <span className={`px-2 py-0.5 rounded text-xs ${s.cls}`}>{s.label}</span>
            {run.status === 'running' && (
              <span className="px-2 py-0.5 rounded text-xs bg-purple-100 text-purple-700">{phaseLabel}</span>
            )}
            {run.status === 'completed' && run.phase && run.phase !== 'done' && (
              <span className="px-2 py-0.5 rounded text-xs bg-amber-100 text-amber-700">{phaseLabel}</span>
            )}
          </div>
          <div className="mt-1 text-xs text-slate-400 flex flex-wrap gap-x-4 gap-y-0.5">
            <span>数据集：<span className="text-slate-600">{run.dataset_name}</span></span>
            <span>candidate：<span className="text-slate-600">{run.candidate_name}</span></span>
            {run.baseline_name && <span>baseline：<span className="text-slate-600">{run.baseline_name}</span></span>}
            <span>judge：<span className="text-slate-600">{run.judge_name}</span></span>
            <span>{run.created_at?.slice(0, 16).replace('T', ' ')}</span>
          </div>

          {/* progress */}
          {run.progress_total > 0 && (
            <div className="mt-2">
              <div className="flex justify-between text-xs text-slate-400 mb-1">
                <span>{run.progress_done} / {run.progress_total} 样本（当前阶段）</span><span>{pct}%</span>
              </div>
              <div className="h-1.5 bg-slate-100 rounded-full overflow-hidden">
                <div className={`h-full rounded-full ${run.status === 'running' ? 'bg-blue-500' : 'bg-slate-400'}`}
                  style={{ width: `${pct}%` }} />
              </div>
            </div>
          )}

          {/* score summary */}
          {(cScore != null || bScore != null) && (
            <div className="mt-3 flex items-center gap-3 text-xs flex-wrap">
              {cScore != null && (
                <div className="px-3 py-1.5 bg-blue-50 text-blue-700 rounded-lg">
                  Candidate&nbsp;<span className="font-bold text-base">{cScore.toFixed(2)}</span>
                  <span className="text-slate-400"> /5</span>
                  {run.candidate_pass_rate != null && (
                    <span className="ml-2 text-slate-500">合格率 {(run.candidate_pass_rate * 100).toFixed(1)}%</span>
                  )}
                </div>
              )}
              {bScore != null && (
                <div className="px-3 py-1.5 bg-amber-50 text-amber-700 rounded-lg">
                  Baseline&nbsp;<span className="font-bold text-base">{bScore.toFixed(2)}</span>
                  <span className="text-slate-400"> /5</span>
                  {run.baseline_pass_rate != null && (
                    <span className="ml-2 text-slate-500">合格率 {(run.baseline_pass_rate * 100).toFixed(1)}%</span>
                  )}
                </div>
              )}
              {delta != null && (
                <div className={`px-2 py-1 rounded text-sm font-bold ${delta > 0 ? 'bg-emerald-100 text-emerald-700' : delta < 0 ? 'bg-red-100 text-red-700' : 'bg-slate-100 text-slate-600'}`}>
                  Δ {delta > 0 ? '+' : ''}{delta.toFixed(2)}
                </div>
              )}
            </div>
          )}

          {run.error_message && (
            <p className="mt-1.5 text-xs text-red-500 break-all">⚠ {run.error_message.slice(0, 200)}</p>
          )}
        </div>

        <div className="flex items-center gap-1.5 shrink-0">
          {run.status === 'running' && (
            <button onClick={handleCancel}
              className="flex items-center gap-1 px-2.5 py-1.5 text-xs bg-amber-500 text-white rounded-lg hover:bg-amber-600">
              <Square size={12} /> 取消
            </button>
          )}
          {(run.status === 'failed' || run.status === 'cancelled' || run.status === 'completed') && (
            <>
              <button onClick={handleResume}
                title="保留已生成回答与评分，仅补未完成部分；切换 vllm 模型后用此按钮"
                className="flex items-center gap-1 px-2.5 py-1.5 text-xs border border-blue-200 text-blue-700 bg-blue-50 rounded-lg hover:bg-blue-100">
                <Play size={12} /> 续跑
              </button>
              <button onClick={handleFreshRestart}
                title="清空所有结果并完全重新评估"
                className="flex items-center gap-1 px-2.5 py-1.5 text-xs border border-slate-200 text-slate-600 rounded-lg hover:bg-slate-50">
                <RefreshCw size={12} /> 全新重跑
              </button>
            </>
          )}
          <button onClick={() => onView(run)}
            className="flex items-center gap-1 px-2.5 py-1.5 text-xs border border-slate-200 rounded-lg hover:bg-slate-50">
            <Eye size={12} /> 明细
          </button>
          {run.status !== 'running' && (
            <button onClick={handleDelete}
              className="p-1.5 text-slate-400 hover:text-red-500 border border-slate-200 rounded-lg hover:bg-red-50">
              <Trash2 size={13} />
            </button>
          )}
        </div>
      </div>
    </div>
  )
}

function EvaluationsTab() {
  const [runs, setRuns] = useState([])
  const [dsList, setDsList] = useState([])
  const [aList, setAList] = useState([])
  const [showCreate, setShowCreate] = useState(false)
  const [viewing, setViewing] = useState(null)
  const [loading, setLoading] = useState(false)

  const load = useCallback(() => {
    setLoading(true)
    Promise.all([
      evaluations.list(),
      datasets.list(),
      assistantsApi.list(),
    ]).then(([r, d, a]) => {
      setRuns(r.data); setDsList(d.data); setAList(a.data)
    }).catch(() => {}).finally(() => setLoading(false))
  }, [])
  useEffect(() => { load() }, [load])
  // poll while any running
  useEffect(() => {
    if (runs.some(r => r.status === 'running')) {
      const t = setInterval(() => evaluations.list().then(r => setRuns(r.data)).catch(() => {}), 4000)
      return () => clearInterval(t)
    }
  }, [runs])

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <p className="text-sm text-slate-500">在测试集上用 LLM-as-Judge 对比 candidate / baseline 模型的事实正确性</p>
        <div className="flex gap-2">
          <button onClick={load}
            className="flex items-center gap-1.5 px-3 py-2 text-sm border border-slate-200 rounded-lg hover:bg-slate-50">
            <RefreshCw size={14} className={loading ? 'animate-spin' : ''} /> 刷新
          </button>
          <button onClick={() => setShowCreate(true)}
            className="flex items-center gap-1.5 px-3 py-2 text-sm bg-blue-600 text-white rounded-lg hover:bg-blue-700">
            <Plus size={14} /> 新建评估
          </button>
        </div>
      </div>

      <div className="space-y-3">
        {runs.map(r => <EvalRunCard key={r.id} run={r} onChanged={load} onView={setViewing} />)}
        {!loading && runs.length === 0 && (
          <div className="text-center py-16 text-slate-400">
            <Award size={36} className="mx-auto mb-3 opacity-30" />
            <p>暂无评估任务，点击「新建评估」开始</p>
            <p className="text-xs mt-1 text-slate-300">建议先在「数据集」页用切分功能造一个小型测试集</p>
          </div>
        )}
      </div>

      {showCreate && (
        <CreateEvalModal datasetList={dsList} assistantList={aList}
          onClose={() => setShowCreate(false)} onCreated={load} />
      )}
      {viewing && <EvaluationItemsModal run={viewing} onClose={() => setViewing(null)} />}
    </div>
  )
}

// ─────────────────────────────── Main Page ───────────────────────────────────

export default function Training() {
  const [tab, setTab] = useState('experiments')
  const [exps, setExps] = useState([])
  const [showCreate, setShowCreate] = useState(false)
  const [loading, setLoading] = useState(false)
  const [stats, setStats] = useState(null)
  const [gpus, setGpus] = useState([])

  const load = async () => {
    setLoading(true)
    try {
      const [expRes, statRes] = await Promise.all([
        training.listExperiments(),
        training.stats(),
      ])
      setExps(expRes.data)
      setStats(statRes.data)
    } catch {}
    setLoading(false)
  }

  useEffect(() => {
    load()
    training.gpuInfo().then(r => setGpus(r.data.gpus || [])).catch(() => {})
  }, [])

  return (
    <div className="p-8 max-w-5xl mx-auto">
      {/* Header */}
      <div className="mb-5">
        <h1 className="text-xl font-bold text-slate-800">训练监控</h1>
        <p className="text-sm text-slate-500 mt-0.5">
          增量预训练 (CPT) → 监督微调 (SFT) → LLM-as-Judge 评估
        </p>
      </div>

      {/* Tabs */}
      <div className="border-b border-slate-200 mb-5">
        <div className="flex gap-1">
          {[
            ['pretrain', '增量预训练 CPT'],
            ['experiments', '微调训练 SFT'],
            ['evaluations', '模型评估'],
          ].map(([k, label]) => (
            <button key={k} onClick={() => setTab(k)}
              className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors -mb-px
                ${tab === k ? 'border-blue-600 text-blue-700' : 'border-transparent text-slate-500 hover:text-slate-700'}`}>
              {label}
            </button>
          ))}
        </div>
      </div>

      {tab === 'pretrain' && <PretrainingTab gpuInfo={gpus} />}

      {tab === 'experiments' && (
        <>
          <div className="flex items-center justify-end mb-5 gap-2">
            <button onClick={load}
              className="flex items-center gap-1.5 px-3 py-2 text-sm border border-slate-200 rounded-lg hover:bg-slate-50">
              <RefreshCw size={14} className={loading ? 'animate-spin' : ''} /> 刷新
            </button>
            <button onClick={() => setShowCreate(true)}
              className="flex items-center gap-1.5 px-3 py-2 text-sm bg-blue-600 text-white rounded-lg hover:bg-blue-700">
              <Plus size={14} /> 新建实验
            </button>
          </div>

          {/* Stats row */}
          {stats && (
            <div className="grid grid-cols-4 gap-3 mb-5">
              {[
                ['全部实验', stats.total, 'text-slate-700'],
                ['训练中', stats.running, 'text-blue-600'],
                ['已完成', stats.completed, 'text-green-600'],
                ['失败', stats.failed, 'text-red-500'],
              ].map(([label, val, cls]) => (
                <div key={label} className="bg-white rounded-xl border border-slate-100 px-4 py-3 shadow-sm text-center">
                  <p className={`text-2xl font-bold ${cls}`}>{val}</p>
                  <p className="text-xs text-slate-400 mt-0.5">{label}</p>
                </div>
              ))}
            </div>
          )}

          {/* GPU Banner */}
          <GpuBanner />

          {/* Experiment List */}
          <div className="space-y-4">
            {exps.map(exp => (
              <ExpCard key={exp.id} exp={exp} onRefresh={load} />
            ))}
            {!loading && exps.length === 0 && (
              <div className="text-center py-16 text-slate-400">
                <BarChart2 size={40} className="mx-auto mb-3 opacity-30" />
                <p>暂无训练实验，点击「新建实验」开始</p>
              </div>
            )}
          </div>

          {showCreate && <CreateExpModal onClose={() => setShowCreate(false)} onCreated={load} />}
        </>
      )}

      {tab === 'evaluations' && <EvaluationsTab />}
    </div>
  )
}
