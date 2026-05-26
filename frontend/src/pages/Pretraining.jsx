import { useEffect, useRef, useState, useCallback } from 'react'
import { pretraining } from '../api/client'
import {
  Plus, RefreshCw, Play, Square, Trash2, ChevronDown, ChevronRight,
  BookOpen, Terminal, AlertCircle, Loader2, Database,
} from 'lucide-react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer,
} from 'recharts'

const STATUS_CFG = {
  pending:   { label: '待启动', cls: 'bg-slate-100 text-slate-600' },
  running:   { label: '预训练中', cls: 'bg-blue-100 text-blue-700' },
  completed: { label: '已完成', cls: 'bg-green-100 text-green-700' },
  failed:    { label: '失败', cls: 'bg-red-100 text-red-700' },
  stopped:   { label: '已停止', cls: 'bg-amber-100 text-amber-700' },
}

const DEFAULT_CFG = {
  learning_rate: 5e-5,
  num_epochs: 1,
  batch_size: 1,
  gradient_accumulation_steps: 16,
  block_size: 2048,
  warmup_ratio: 0.03,
  weight_decay: 0.01,
  logging_steps: 10,
  eval_steps: 200,
  save_steps: 200,
  use_lora: true,
  lora_r: 16,
  lora_alpha: 32,
  lora_dropout: 0.05,
  lora_target_modules: 'all-linear',
  use_4bit: false,
  use_bf16: true,
}

const DEFAULT_FILTER = {
  document_types: ['case_report', 'guideline'],
  min_content_length: 200,
  doc_limit: '',
  eval_ratio: 0.05,
}

// ─────────────────────────────── Create Modal ────────────────────────────

function CreateCPTModal({ onClose, onCreated, gpuInfo }) {
  const [form, setForm] = useState({
    name: '',
    base_model: '',
    output_dir: '',
  })
  const [filt, setFilt] = useState(DEFAULT_FILTER)
  const [cfg, setCfg] = useState(DEFAULT_CFG)
  const [tab, setTab] = useState('basic')
  const [preview, setPreview] = useState(null)
  const [previewLoading, setPreviewLoading] = useState(false)
  const [loading, setLoading] = useState(false)

  // gpu selection — same UX as Training page
  const [gpuMode, setGpuMode] = useState('auto')
  const [pickedSingle, setPickedSingle] = useState(null)
  const [pickedMulti, setPickedMulti] = useState([])
  const [manualIds, setManualIds] = useState('')

  const set = (k, v) => setForm(f => ({ ...f, [k]: v }))
  const setCfgVal = (k, v) => setCfg(c => ({ ...c, [k]: v }))
  const setFiltVal = (k, v) => setFilt(f => ({ ...f, [k]: v }))

  const resolveGpuIds = () => {
    if (gpuMode === 'auto') return null
    if (gpuMode === 'cpu') return []
    if (gpuInfo.length > 0) {
      if (gpuMode === 'single') return pickedSingle != null ? [pickedSingle] : null
      return pickedMulti.length ? [...pickedMulti].sort((a, b) => a - b) : null
    }
    const ids = manualIds.split(',').map(s => s.trim()).filter(Boolean).map(Number)
      .filter(n => Number.isInteger(n) && n >= 0)
    if (gpuMode === 'single') return ids.length ? [ids[0]] : null
    return ids.length ? ids : null
  }

  const fetchPreview = async () => {
    setPreviewLoading(true)
    try {
      const payload = {
        ...filt,
        doc_limit: filt.doc_limit ? Number(filt.doc_limit) : null,
        eval_ratio: Number(filt.eval_ratio),
      }
      const r = await pretraining.previewCorpus(payload)
      setPreview(r.data)
    } catch (e) {
      setPreview({ error: e?.response?.data?.detail || '预览失败' })
    }
    setPreviewLoading(false)
  }

  const handleSubmit = async () => {
    if (!form.name.trim() || !form.base_model.trim()) return
    setLoading(true)
    try {
      const cf = {
        ...filt,
        doc_limit: filt.doc_limit ? Number(filt.doc_limit) : null,
        eval_ratio: Number(filt.eval_ratio),
      }
      await pretraining.createExperiment({
        name: form.name.trim(),
        base_model: form.base_model.trim(),
        output_dir: form.output_dir.trim() || undefined,
        corpus_filter: cf,
        config: { ...cfg, gpu_ids: resolveGpuIds() },
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
        value={cfg[key]}
        onChange={e => setCfgVal(key, step < 1 ? parseFloat(e.target.value) : parseInt(e.target.value))} />
    </div>
  )

  return (
    <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-2xl max-h-[90vh] flex flex-col">
        <div className="px-6 pt-6 pb-3 border-b border-slate-100">
          <h2 className="text-base font-semibold text-slate-800">新建增量预训练实验</h2>
          <p className="text-xs text-slate-500 mt-1">
            将原始病例 / 指南文档以 CLM (next-token) 方式注入基座模型，用于后续 SFT 微调
          </p>
        </div>

        <div className="flex gap-1 px-6 pt-3 flex-wrap">
          {[['basic','基本'], ['corpus','语料'], ['hparam','超参数'], ['lora','LoRA / 量化'], ['hardware','硬件资源']].map(([k, label]) => (
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
                <label className={labelCls}>实验名称 *</label>
                <input className={inputCls} value={form.name} onChange={e => set('name', e.target.value)}
                  placeholder="例：Qwen2.5-7B 医学语料 CPT v1" />
              </div>
              <div>
                <label className={labelCls}>基座模型路径 *</label>
                <input className={inputCls} value={form.base_model} onChange={e => set('base_model', e.target.value)}
                  placeholder="/data/models/Qwen2.5-7B-Instruct 或 HF model id" />
              </div>
              <div>
                <label className={labelCls}>输出路径（留空自动生成）</label>
                <input className={inputCls} value={form.output_dir} onChange={e => set('output_dir', e.target.value)}
                  placeholder="留空 → ./pretrain_runs/<id>/output" />
                <p className="text-xs text-slate-400 mt-1">
                  CPT 完成后，把输出路径填到 SFT 实验的「基座模型路径」即可继续微调
                </p>
              </div>
            </div>
          )}

          {tab === 'corpus' && (
            <div className="space-y-3">
              <div>
                <label className={labelCls}>文档类型</label>
                <div className="flex gap-3">
                  {[['case_report', '病例报告'], ['guideline', '临床指南']].map(([k, label]) => (
                    <label key={k} className="flex items-center gap-2 text-sm">
                      <input type="checkbox" className="accent-blue-600"
                        checked={filt.document_types.includes(k)}
                        onChange={e => {
                          setFilt(f => ({
                            ...f,
                            document_types: e.target.checked
                              ? [...new Set([...f.document_types, k])]
                              : f.document_types.filter(t => t !== k),
                          }))
                        }} />
                      {label}
                    </label>
                  ))}
                </div>
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className={labelCls}>最小字符数（过滤短/不完整文档）</label>
                  <input type="number" min={0} className={inputCls}
                    value={filt.min_content_length}
                    onChange={e => setFiltVal('min_content_length', parseInt(e.target.value || '0'))} />
                </div>
                <div>
                  <label className={labelCls}>文档数量上限（留空 = 全部）</label>
                  <input type="number" min={1} className={inputCls}
                    value={filt.doc_limit}
                    onChange={e => setFiltVal('doc_limit', e.target.value)}
                    placeholder="例：500（用于先小规模试跑）" />
                </div>
              </div>
              <div>
                <label className={labelCls}>
                  验证集比例：{(filt.eval_ratio * 100).toFixed(1)}%
                  <span className="text-slate-400 ml-1">（用于跟踪 perplexity）</span>
                </label>
                <input type="range" min={0} max={0.2} step={0.005}
                  className="w-full accent-blue-600"
                  value={filt.eval_ratio}
                  onChange={e => setFiltVal('eval_ratio', parseFloat(e.target.value))} />
              </div>

              <div className="border-t border-slate-100 pt-3">
                <button onClick={fetchPreview}
                  className="flex items-center gap-1.5 px-3 py-1.5 text-xs border border-slate-200 rounded-lg hover:bg-slate-50">
                  <Database size={12} /> 预览语料
                  {previewLoading && <Loader2 size={12} className="animate-spin" />}
                </button>
                {preview && (
                  preview.error ? (
                    <p className="mt-2 text-xs text-red-500">{preview.error}</p>
                  ) : (
                    <div className="mt-3 p-3 bg-blue-50 rounded-lg text-xs text-slate-700 space-y-1">
                      <p><b>{preview.doc_count}</b> 篇文档（病例 {preview.by_type?.case_report ?? 0} · 指南 {preview.by_type?.guideline ?? 0}）</p>
                      <p>采样 {preview.sampled} 篇，共 <b>{preview.total_chars_sampled?.toLocaleString()}</b> 字符；
                        预估 <b>{preview.estimated_tokens?.toLocaleString()}</b> tokens</p>
                      <p className="text-slate-500">其中 {preview.eval_split_count} 篇用作验证集</p>
                    </div>
                  )
                )}
              </div>
            </div>
          )}

          {tab === 'hparam' && (
            <div className="grid grid-cols-2 gap-3">
              {numInput('learning_rate', '学习率', 1e-6, 1e-3, 1e-6)}
              {numInput('num_epochs', '训练轮次 (Epoch)', 1, 10)}
              {numInput('batch_size', '每卡批大小', 1, 32)}
              {numInput('gradient_accumulation_steps', '梯度累积步数', 1, 64)}
              {numInput('block_size', '上下文窗口 (tokens)', 512, 32768, 256)}
              {numInput('logging_steps', 'Log 步数间隔', 1, 500)}
              {numInput('eval_steps', '验证步数间隔', 10, 2000)}
              {numInput('save_steps', '保存步数间隔', 10, 5000)}
              {numInput('warmup_ratio', '预热比例', 0, 0.5, 0.01)}
              {numInput('weight_decay', '权重衰减', 0, 0.1, 0.001)}
              <div className="col-span-2 text-xs text-slate-500 px-2 py-2 bg-amber-50 rounded">
                ⚠ CPT 学习率通常显著低于 SFT（5e-5 ~ 1e-4），过大会导致灾难性遗忘
              </div>
            </div>
          )}

          {tab === 'lora' && (
            <div className="space-y-3">
              <label className="flex items-center gap-2">
                <input type="checkbox" checked={cfg.use_lora}
                  onChange={e => setCfgVal('use_lora', e.target.checked)}
                  className="w-4 h-4 accent-blue-600" />
                <span className="text-sm font-medium text-slate-700">启用 LoRA（推荐：节省显存，便于 SFT 阶段叠加）</span>
              </label>
              {cfg.use_lora && (
                <div className="grid grid-cols-2 gap-3 pl-6">
                  {numInput('lora_r', 'LoRA Rank', 4, 256)}
                  {numInput('lora_alpha', 'LoRA Alpha', 8, 512)}
                  {numInput('lora_dropout', 'LoRA Dropout', 0, 0.5, 0.01)}
                  <div>
                    <label className={labelCls}>目标模块</label>
                    <input className={inputCls} value={cfg.lora_target_modules}
                      onChange={e => setCfgVal('lora_target_modules', e.target.value)} />
                  </div>
                </div>
              )}
              <div className="border-t border-slate-100 pt-3 space-y-2">
                <label className="flex items-center gap-2 text-sm">
                  <input type="checkbox" checked={cfg.use_4bit}
                    onChange={e => setCfgVal('use_4bit', e.target.checked)}
                    className="w-4 h-4 accent-blue-600" />
                  4-bit 量化加载（QLoRA）
                </label>
                <label className="flex items-center gap-2 text-sm">
                  <input type="checkbox" checked={cfg.use_bf16}
                    onChange={e => setCfgVal('use_bf16', e.target.checked)}
                    className="w-4 h-4 accent-blue-600" />
                  使用 BF16
                </label>
              </div>
            </div>
          )}

          {tab === 'hardware' && (
            <div className="space-y-3">
              <p className="text-xs text-slate-500">CPT 通常显存压力大于 SFT；多卡 DDP 可线性扩展 batch</p>
              <div className="grid grid-cols-2 gap-2">
                {[
                  ['auto',   '自动',     '不修改 CUDA_VISIBLE_DEVICES'],
                  ['single', '单卡训练', '指定一张 GPU'],
                  ['multi',  '多卡训练', '使用 torchrun + DDP'],
                  ['cpu',    '仅 CPU',  '强制走 CPU（不建议）'],
                ].map(([k, label, desc]) => (
                  <label key={k}
                    className={`flex items-start gap-2 p-2.5 rounded-lg border cursor-pointer transition-colors
                      ${gpuMode === k ? 'border-blue-400 bg-blue-50' : 'border-slate-200 hover:bg-slate-50'}`}>
                    <input type="radio" name="cptGpuMode" className="mt-0.5 accent-blue-600"
                      checked={gpuMode === k} onChange={() => setGpuMode(k)} />
                    <div className="min-w-0">
                      <p className="text-sm font-medium text-slate-700">{label}</p>
                      <p className="text-xs text-slate-400 mt-0.5">{desc}</p>
                    </div>
                  </label>
                ))}
              </div>
              {(gpuMode === 'single' || gpuMode === 'multi') && (
                <div className="border-t border-slate-100 pt-3">
                  {gpuInfo.length > 0 ? (
                    <div className="space-y-1.5">
                      {gpuInfo.map(g => {
                        const isPicked = gpuMode === 'single'
                          ? pickedSingle === g.index
                          : pickedMulti.includes(g.index)
                        return (
                          <label key={g.index}
                            className={`flex items-center gap-3 p-2 rounded-lg border cursor-pointer text-xs
                              ${isPicked ? 'border-blue-400 bg-blue-50' : 'border-slate-200'}`}>
                            <input type={gpuMode === 'single' ? 'radio' : 'checkbox'}
                              name="cptGpuPick" className="accent-blue-600"
                              checked={isPicked}
                              onChange={() => {
                                if (gpuMode === 'single') setPickedSingle(g.index)
                                else setPickedMulti(p => p.includes(g.index)
                                  ? p.filter(x => x !== g.index) : [...p, g.index])
                              }} />
                            <div className="flex-1 min-w-0">
                              <p className="text-slate-700">
                                <span className="font-medium">GPU {g.index}</span>
                                <span className="ml-2 text-slate-500">{g.name}</span>
                              </p>
                              <p className="text-[10px] text-slate-400 mt-0.5">
                                空闲 {g.memory_free_mb} MB / {g.memory_total_mb} MB
                              </p>
                            </div>
                          </label>
                        )
                      })}
                    </div>
                  ) : (
                    <input type="text" value={manualIds}
                      onChange={e => setManualIds(e.target.value)}
                      placeholder={gpuMode === 'single' ? '例：0' : '例：0,1'}
                      className={inputCls} />
                  )}
                </div>
              )}
              <p className="text-[11px] text-slate-400">
                最终下发：<code className="px-1.5 py-0.5 bg-slate-100 rounded">gpu_ids = {(() => {
                  const v = resolveGpuIds()
                  if (v === null) return 'null （auto）'
                  if (v.length === 0) return '[] （CPU）'
                  return `[${v.join(', ')}]`
                })()}</code>
              </p>
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

// ─────────────────────────────── Metrics chart ────────────────────────────

function CPTMetricsPanel({ expId }) {
  const [metrics, setMetrics] = useState([])
  const [activeChart, setActiveChart] = useState('loss')

  useEffect(() => {
    const poll = () => pretraining.getMetrics(expId).then(r => setMetrics(r.data)).catch(() => {})
    poll()
    const t = setInterval(poll, 4000)
    return () => clearInterval(t)
  }, [expId])

  if (metrics.length === 0) {
    return <p className="text-xs text-slate-400 py-6 text-center">暂无指标，启动后自动更新</p>
  }

  const hasEval = metrics.some(m => m.eval_loss != null)
  const hasPpl = metrics.some(m => m.perplexity != null)
  const hasLr = metrics.some(m => m.learning_rate != null)
  const hasGrad = metrics.some(m => m.extra_metrics?.grad_norm != null)

  const chartData = metrics.map(m => ({
    step: m.step,
    train_loss: m.train_loss,
    eval_loss: m.eval_loss,
    perplexity: m.perplexity,
    learning_rate: m.learning_rate,
    grad_norm: m.extra_metrics?.grad_norm,
  }))

  const tabs = [
    { key: 'loss', label: 'Loss' },
    ...(hasPpl ? [{ key: 'ppl', label: 'Perplexity' }] : []),
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
        <span className="ml-auto text-xs text-slate-400 self-center">{metrics.length} 数据点</span>
      </div>
      <div className="h-48">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={{ top: 4, right: 16, left: -10, bottom: 4 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
            <XAxis dataKey="step" tick={{ fontSize: 10 }} />
            <YAxis tick={{ fontSize: 10 }} domain={['auto', 'auto']} />
            <Tooltip contentStyle={{ fontSize: 11 }} formatter={(v) => typeof v === 'number' ? v.toFixed(6) : v} />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            {activeChart === 'loss' && metrics.some(m => m.train_loss != null) && (
              <Line type="monotone" dataKey="train_loss" stroke="#3b82f6" dot={false}
                name="训练损失" strokeWidth={1.5} connectNulls isAnimationActive={false} />
            )}
            {activeChart === 'loss' && hasEval && (
              <Line type="monotone" dataKey="eval_loss" stroke="#10b981" dot={false}
                name="验证损失" strokeWidth={1.5} connectNulls isAnimationActive={false} />
            )}
            {activeChart === 'ppl' && (
              <Line type="monotone" dataKey="perplexity" stroke="#10b981" dot={false}
                name="Perplexity" strokeWidth={1.5} connectNulls isAnimationActive={false} />
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

// ─────────────────────────────── Log console ──────────────────────────────

const LOG_LEVEL_CLS = {
  info:    'text-slate-300',
  metrics: 'text-green-400',
  warning: 'text-amber-400',
  error:   'text-red-400',
}

function CPTLogConsole({ expId, isRunning }) {
  const [logs, setLogs] = useState([])
  const [connected, setConnected] = useState(false)
  const bottomRef = useRef(null)
  const esRef = useRef(null)
  const logsRef = useRef([])
  const lastIdRef = useRef(0)
  const scroll = useCallback(() => bottomRef.current?.scrollIntoView({ behavior: 'smooth' }), [])

  useEffect(() => {
    pretraining.getLogs(expId, { limit: 300 }).then(r => {
      logsRef.current = r.data
      setLogs([...r.data])
      if (r.data.length > 0) lastIdRef.current = r.data[r.data.length - 1].id
      setTimeout(scroll, 100)
    }).catch(() => {})
  }, [expId, scroll])

  useEffect(() => {
    if (!isRunning) { setConnected(false); return }
    const connect = () => {
      const url = pretraining.logsStreamUrl(expId, lastIdRef.current)
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
          scroll()
        } catch {}
      }
      es.addEventListener('done', () => { setConnected(false); es.close() })
      es.onerror = () => { setConnected(false); es.close(); setTimeout(connect, 3000) }
    }
    connect()
    return () => esRef.current?.close()
  }, [expId, isRunning, scroll])

  const fmt = (msg) => {
    try {
      const d = JSON.parse(msg)
      if (d.type === 'metrics') {
        const parts = []
        if (d.step != null) parts.push(`step=${d.step}`)
        if (d.epoch != null) parts.push(`epoch=${Number(d.epoch).toFixed(2)}`)
        if (d.train_loss != null) parts.push(`train_loss=${Number(d.train_loss).toFixed(4)}`)
        if (d.eval_loss != null) parts.push(`eval_loss=${Number(d.eval_loss).toFixed(4)}`)
        if (d.perplexity != null) parts.push(`ppl=${Number(d.perplexity).toFixed(2)}`)
        if (d.learning_rate) parts.push(`lr=${Number(d.learning_rate).toExponential(3)}`)
        return `[metrics] ${parts.join('  ')}`
      }
      if (d.type === 'progress') return `[progress] ${d.message}`
      if (d.type === 'error') return `[error] ${d.message}`
      if (d.type === 'info') return `[info] ${JSON.stringify(d)}`
      if (d.type === 'completed') return `[completed] 已保存：${d.output_dir}`
      if (d.type === 'final_eval') return `[final_eval] eval_loss=${d.eval_loss?.toFixed(4)} ppl=${d.perplexity?.toFixed(2)}`
      return msg
    } catch {
      return msg
    }
  }

  return (
    <div className="mt-3">
      <div className="flex items-center gap-2 mb-1.5">
        <Terminal size={13} className="text-slate-400" />
        <span className="text-xs font-medium text-slate-500">CPT 日志</span>
        {isRunning && (
          <span className={`flex items-center gap-1 text-xs ml-auto ${connected ? 'text-green-500' : 'text-amber-500'}`}>
            <span className={`w-1.5 h-1.5 rounded-full ${connected ? 'bg-green-500 animate-pulse' : 'bg-amber-400'}`} />
            {connected ? '实时连接' : '重连中...'}
          </span>
        )}
      </div>
      <div className="bg-slate-900 rounded-xl p-3 h-52 overflow-y-auto font-mono text-xs leading-relaxed">
        {logs.length === 0 && <span className="text-slate-500">等待输出...</span>}
        {logs.map((lg, i) => (
          <div key={lg.id ?? i} className={`${LOG_LEVEL_CLS[lg.level] ?? 'text-slate-300'} whitespace-pre-wrap break-all`}>
            {fmt(lg.message)}
          </div>
        ))}
        <div ref={bottomRef} />
      </div>
    </div>
  )
}

// ─────────────────────────────── Card ─────────────────────────────────────

function ProgressBar({ current, total }) {
  if (!total) return null
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

function CPTCard({ exp, onRefresh }) {
  const [expanded, setExpanded] = useState(false)
  const [activeTab, setActiveTab] = useState('metrics')
  const [acting, setActing] = useState(false)
  const [data, setData] = useState(exp)
  const pollRef = useRef(null)

  useEffect(() => {
    if (data.status === 'running') {
      pollRef.current = setInterval(() => {
        pretraining.getExperiment(data.id).then(r => setData(r.data)).catch(() => {})
      }, 3000)
    }
    return () => clearInterval(pollRef.current)
  }, [data.status, data.id])

  const handleStart = async () => {
    setActing(true)
    try {
      await pretraining.start(data.id)
      await pretraining.getExperiment(data.id).then(r => setData(r.data))
    } catch (e) { alert(e?.response?.data?.detail || '启动失败') }
    setActing(false)
  }
  const handleStop = async () => {
    if (!confirm('停止 CPT？')) return
    setActing(true)
    try {
      await pretraining.stop(data.id)
      await pretraining.getExperiment(data.id).then(r => setData(r.data))
    } catch {}
    setActing(false)
  }
  const handleDelete = async () => {
    if (!confirm(`删除实验「${data.name}」？`)) return
    try { await pretraining.deleteExperiment(data.id); onRefresh() }
    catch (e) { alert(e?.response?.data?.detail || '删除失败') }
  }

  const s = STATUS_CFG[data.status] || STATUS_CFG.pending
  const cfg = data.config || {}
  const isRunning = data.status === 'running'

  const gpuLabel = (() => {
    const ids = cfg.gpu_ids
    if (ids === undefined || ids === null) return '自动'
    if (Array.isArray(ids) && ids.length === 0) return '仅 CPU'
    if (Array.isArray(ids) && ids.length === 1) return `单卡 GPU ${ids[0]}`
    if (Array.isArray(ids)) return `多卡 DDP [${ids.join(',')}]`
    return String(ids)
  })()

  const cs = data.corpus_stats || cfg.corpus_export
  const fe = data.final_eval

  return (
    <div className="bg-white rounded-2xl shadow-sm border border-slate-100 overflow-hidden">
      <div className="px-5 py-4">
        <div className="flex items-start justify-between gap-3">
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 flex-wrap">
              <span className="font-semibold text-slate-800 truncate">{data.name}</span>
              <span className={`px-2 py-0.5 rounded-full text-xs font-medium shrink-0 ${s.cls}`}>
                {isRunning && <span className="inline-block w-1.5 h-1.5 rounded-full bg-blue-500 animate-pulse mr-1 align-middle" />}
                {s.label}
              </span>
              <span className="px-2 py-0.5 rounded text-xs bg-purple-100 text-purple-700">CPT</span>
            </div>
            <div className="mt-1 text-xs text-slate-400 flex flex-wrap gap-x-4 gap-y-0.5">
              <span>基座：{data.base_model}</span>
              {cs && (
                <span>语料：{cs.train_blocks ?? cs.train_docs ?? 0} 块 / {cs.total_tokens?.toLocaleString() ?? '?'} tokens</span>
              )}
              {data.best_eval_loss != null && (
                <span className="text-green-600 font-medium">最优 eval_loss：{data.best_eval_loss.toFixed(4)}</span>
              )}
              {fe?.perplexity != null && (
                <span className="text-emerald-600 font-medium">最终 ppl：{fe.perplexity.toFixed(2)}</span>
              )}
              {cfg.use_lora && <span className="text-purple-600">LoRA r={cfg.lora_r}</span>}
              {cfg.use_4bit && <span className="text-orange-500">4-bit</span>}
              <span className="text-blue-600">{gpuLabel}</span>
              <span>创建：{data.created_at?.slice(0, 10)}</span>
            </div>
            {isRunning && <ProgressBar current={data.current_step} total={data.total_steps} />}
            {data.error_message && (
              <p className="mt-1 text-xs text-red-500 flex items-center gap-1 break-all">
                <AlertCircle size={11} className="shrink-0" /> {data.error_message.slice(0, 200)}
              </p>
            )}
            {data.final_output_dir && data.status === 'completed' && (
              <div className="mt-2 px-2.5 py-1.5 bg-emerald-50 rounded text-xs text-emerald-700 flex items-center gap-2">
                <BookOpen size={12} className="shrink-0" />
                可用作 SFT 基座：<code className="break-all">{data.final_output_dir}</code>
              </div>
            )}
          </div>

          <div className="flex items-center gap-1.5 shrink-0">
            {(data.status === 'pending' || data.status === 'stopped' || data.status === 'failed') && (
              <button onClick={handleStart} disabled={acting}
                className="flex items-center gap-1 px-3 py-1.5 text-xs bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50">
                {acting ? <Loader2 size={12} className="animate-spin" /> : <Play size={12} />}
                {data.status === 'pending' ? '启动 CPT' : '重新训练'}
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
              {expanded ? '收起' : '详情'}
              {expanded ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
            </button>
            {!isRunning && (
              <button onClick={handleDelete}
                className="p-1.5 text-slate-400 hover:text-red-500 hover:bg-red-50 rounded-lg">
                <Trash2 size={14} />
              </button>
            )}
          </div>
        </div>
      </div>

      {expanded && (
        <div className="border-t border-slate-100 px-5 pb-5">
          <div className="flex gap-1 pt-3 mb-2">
            {[['metrics', '指标曲线'], ['logs', 'CPT 日志'], ['config', '实验配置']].map(([k, label]) => (
              <button key={k} onClick={() => setActiveTab(k)}
                className={`px-3 py-1.5 text-xs rounded-lg font-medium transition-colors
                  ${activeTab === k ? 'bg-blue-100 text-blue-700' : 'text-slate-400 hover:bg-slate-100'}`}>
                {label}
              </button>
            ))}
          </div>
          {activeTab === 'metrics' && <CPTMetricsPanel expId={data.id} />}
          {activeTab === 'logs' && <CPTLogConsole expId={data.id} isRunning={isRunning} />}
          {activeTab === 'config' && (
            <div className="mt-2 bg-slate-50 rounded-xl p-4">
              <dl className="grid grid-cols-2 gap-x-6 gap-y-1.5 text-xs">
                {[
                  ['基座模型', data.base_model],
                  ['权重保存目录', data.final_output_dir || cfg.output_dir || '自动'],
                  ['文档类型', (data.corpus_filter?.document_types || []).join(' / ') || '-'],
                  ['最小字符数', data.corpus_filter?.min_content_length ?? '-'],
                  ['文档数量上限', data.corpus_filter?.doc_limit ?? '全部'],
                  ['验证集比例', `${((data.corpus_filter?.eval_ratio ?? 0) * 100).toFixed(1)}%`],
                  ['上下文长度', cfg.block_size],
                  ['学习率', cfg.learning_rate],
                  ['Epoch', cfg.num_epochs],
                  ['批大小', cfg.batch_size],
                  ['梯度累积', cfg.gradient_accumulation_steps],
                  ['LoRA', cfg.use_lora ? `r=${cfg.lora_r} α=${cfg.lora_alpha}` : '关闭'],
                  ['4-bit', cfg.use_4bit ? '是' : '否'],
                  ['精度', cfg.use_bf16 ? 'BF16' : 'FP16'],
                  ['GPU', gpuLabel],
                  ['最优 eval_loss', data.best_eval_loss?.toFixed(6) ?? '-'],
                  ['最终 perplexity',
                    fe?.perplexity != null ? fe.perplexity.toFixed(4)
                      : (data.status === 'completed' ? '未生成（无验证集？）' : '训练完成后生成')],
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

// ─────────────────────────────── Top-level Tab ───────────────────────────

export default function PretrainingTab({ gpuInfo }) {
  const [exps, setExps] = useState([])
  const [stats, setStats] = useState(null)
  const [showCreate, setShowCreate] = useState(false)
  const [loading, setLoading] = useState(false)

  const load = async () => {
    setLoading(true)
    try {
      const [eRes, sRes] = await Promise.all([
        pretraining.listExperiments(),
        pretraining.stats(),
      ])
      setExps(eRes.data)
      setStats(sRes.data)
    } catch {}
    setLoading(false)
  }

  useEffect(() => { load() }, [])

  return (
    <div>
      <div className="flex items-center justify-end mb-5 gap-2">
        <button onClick={load}
          className="flex items-center gap-1.5 px-3 py-2 text-sm border border-slate-200 rounded-lg hover:bg-slate-50">
          <RefreshCw size={14} className={loading ? 'animate-spin' : ''} /> 刷新
        </button>
        <button onClick={() => setShowCreate(true)}
          className="flex items-center gap-1.5 px-3 py-2 text-sm bg-blue-600 text-white rounded-lg hover:bg-blue-700">
          <Plus size={14} /> 新建 CPT 实验
        </button>
      </div>

      {stats && (
        <div className="grid grid-cols-4 gap-3 mb-5">
          {[
            ['全部实验', stats.total, 'text-slate-700'],
            ['进行中', stats.running, 'text-blue-600'],
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

      <div className="space-y-4">
        {exps.map(exp => <CPTCard key={exp.id} exp={exp} onRefresh={load} />)}
        {!loading && exps.length === 0 && (
          <div className="text-center py-16 text-slate-400">
            <BookOpen size={40} className="mx-auto mb-3 opacity-30" />
            <p>暂无 CPT 实验</p>
            <p className="text-xs mt-1 text-slate-300">在 SFT 微调前先做增量预训练，注入医学领域知识</p>
          </div>
        )}
      </div>

      {showCreate && (
        <CreateCPTModal gpuInfo={gpuInfo}
          onClose={() => setShowCreate(false)} onCreated={load} />
      )}
    </div>
  )
}
