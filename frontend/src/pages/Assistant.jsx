import { useEffect, useState, useCallback } from 'react'
import { assistant, assistants as assistantsApi, training } from '../api/client'
import {
  Plus, RefreshCw, Play, Square, Trash2, Edit3, Server, Cloud,
  Send, Stethoscope, Loader2, AlertCircle, CheckCircle2, Clock, ChevronDown, ChevronUp,
} from 'lucide-react'
import MarkdownView from '../components/MarkdownView'

const STATUS_CFG = {
  stopped:  { label: '已停止', cls: 'bg-slate-100 text-slate-600',  icon: Square },
  starting: { label: '启动中', cls: 'bg-amber-100 text-amber-700',  icon: Loader2 },
  running:  { label: '运行中', cls: 'bg-green-100 text-green-700',  icon: CheckCircle2 },
  failed:   { label: '失败',  cls: 'bg-red-100 text-red-700',      icon: AlertCircle },
}

// ─────────────────────────────── Create / Edit Modal ───────────────────────────────

function AssistantFormModal({ initial, experiments, onClose, onSaved }) {
  const isEdit = !!initial
  const [form, setForm] = useState(initial ? {
    ...initial,
    extra_vllm_args: Array.isArray(initial.extra_vllm_args) ? initial.extra_vllm_args.join(' ') : '',
    api_key: '',  // never pre-fill secrets
  } : {
    name: '',
    type: 'remote',                    // remote | local
    description: '',
    base_url: '',
    model_name: '',
    api_key: '',
    model_path: '',
    max_model_len: 16384,
    extra_vllm_args: '',
    lora_adapter_path: '',
    source_experiment_id: '',
    gpu_ids: null,
  })
  const [loading, setLoading] = useState(false)
  const [err, setErr] = useState('')
  const [showKey, setShowKey] = useState(false)

  // GPU selection (mirrors training-page pattern)
  // gpuMode: 'auto' | 'single' | 'multi'
  const initialGpuMode = (() => {
    const ids = initial?.gpu_ids
    if (!Array.isArray(ids) || ids.length === 0) return 'auto'
    return ids.length === 1 ? 'single' : 'multi'
  })()
  const [gpuMode, setGpuMode] = useState(initialGpuMode)
  const [availableGpus, setAvailableGpus] = useState([])
  const [gpuLoading, setGpuLoading] = useState(false)
  const [pickedSingle, setPickedSingle] = useState(
    Array.isArray(initial?.gpu_ids) && initial.gpu_ids.length === 1 ? initial.gpu_ids[0] : null
  )
  const [pickedMulti, setPickedMulti] = useState(
    Array.isArray(initial?.gpu_ids) && initial.gpu_ids.length > 1 ? [...initial.gpu_ids] : []
  )
  const [manualIds, setManualIds] = useState(
    Array.isArray(initial?.gpu_ids) ? initial.gpu_ids.join(',') : ''
  )

  const refreshGpus = useCallback(() => {
    setGpuLoading(true)
    assistantsApi.gpuInfo()
      .then(r => setAvailableGpus(r.data.gpus || []))
      .catch(() => setAvailableGpus([]))
      .finally(() => setGpuLoading(false))
  }, [])

  useEffect(() => {
    refreshGpus()
  }, [refreshGpus])

  const set = (k, v) => setForm(f => ({ ...f, [k]: v }))

  const fillFromExperiment = (expId) => {
    const exp = experiments.find(e => e.id === Number(expId))
    if (!exp) return
    const cfg = exp.config || {}
    const useLora = !!cfg.use_lora
    const outputDir = exp.final_output_dir || cfg.output_dir || ''
    setForm(f => ({
      ...f,
      type: 'local',
      source_experiment_id: expId,
      name: f.name || `exp-${exp.id}-${exp.name.slice(0, 20)}`,
      model_name: f.model_name || `exp-${exp.id}-${exp.name}`.replace(/\s+/g, '_'),
      model_path: useLora ? exp.base_model : outputDir,
      lora_adapter_path: useLora ? outputDir : '',
      max_model_len: f.max_model_len || cfg.max_seq_length || 16384,
      description: f.description || `由训练实验 #${exp.id} (${exp.name}) 生成`,
    }))
  }

  // Resolve gpuMode + selections → gpu_ids payload (null | int[])
  const resolveGpuIds = () => {
    if (gpuMode === 'auto') return null
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
    setErr('')
    if (!form.name?.trim()) { setErr('请输入名称'); return }
    if (!form.model_name?.trim()) { setErr('请输入 model_name'); return }
    if (form.type === 'remote') {
      if (!form.base_url?.trim()) { setErr('远程助手必须填写 base_url'); return }
    } else {
      if (!form.model_path?.trim()) { setErr('本地助手必须填写 model_path'); return }
    }
    setLoading(true)
    try {
      const payload = {
        name: form.name.trim(),
        type: form.type,
        description: form.description?.trim() || undefined,
        model_name: form.model_name.trim(),
      }
      if (form.type === 'remote') {
        payload.base_url = form.base_url.trim()
        payload.api_key = form.api_key?.trim() || undefined
      } else {
        payload.model_path = form.model_path.trim()
        if (form.max_model_len) payload.max_model_len = Number(form.max_model_len)
        if (form.lora_adapter_path?.trim()) payload.lora_adapter_path = form.lora_adapter_path.trim()
        if (form.extra_vllm_args?.trim()) {
          payload.extra_vllm_args = form.extra_vllm_args.split(/\s+/).filter(Boolean)
        } else {
          payload.extra_vllm_args = []
        }
        if (form.source_experiment_id) payload.source_experiment_id = Number(form.source_experiment_id)
        // Always send gpu_ids; null means auto. Use [] sentinel only for the
        // narrow CPU case (vllm doesn't support CPU well, so we never send []).
        payload.gpu_ids = resolveGpuIds()
      }
      if (isEdit) {
        await assistantsApi.update(initial.id, payload)
      } else {
        await assistantsApi.create(payload)
      }
      onSaved()
      onClose()
    } catch (e) {
      setErr(e?.response?.data?.detail || '保存失败')
    }
    setLoading(false)
  }

  const inputCls = 'w-full px-3 py-2 text-sm border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500'
  const labelCls = 'text-xs text-slate-500 mb-1 block'

  return (
    <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-xl max-h-[90vh] flex flex-col">
        <div className="px-6 pt-6 pb-3 border-b border-slate-100">
          <h2 className="text-base font-semibold text-slate-800">
            {isEdit ? `编辑助手：${initial.name}` : '新建助手'}
          </h2>
        </div>

        <div className="overflow-y-auto flex-1 px-6 py-4 space-y-3">
          {!isEdit && (
            <div className="grid grid-cols-2 gap-2">
              {[
                ['remote', '远程助手', '调用 OpenAI 兼容 API（如 OpenAI、Claude、DeepSeek 等）'],
                ['local',  '本地助手', '后端用 vllm serve 启动指定模型路径，提供 OpenAI 兼容服务'],
              ].map(([k, label, desc]) => (
                <label key={k}
                  className={`flex items-start gap-2 p-3 rounded-lg border cursor-pointer transition-colors
                    ${form.type === k ? 'border-blue-400 bg-blue-50' : 'border-slate-200 hover:bg-slate-50'}`}>
                  <input type="radio" name="atype" className="mt-0.5 accent-blue-600"
                    checked={form.type === k} onChange={() => set('type', k)} />
                  <div className="min-w-0">
                    <p className="text-sm font-medium text-slate-700 flex items-center gap-1">
                      {k === 'remote' ? <Cloud size={13} /> : <Server size={13} />}{label}
                    </p>
                    <p className="text-xs text-slate-400 mt-0.5 leading-relaxed">{desc}</p>
                  </div>
                </label>
              ))}
            </div>
          )}

          {form.type === 'local' && !isEdit && experiments?.length > 0 && (
            <div className="p-3 bg-blue-50 border border-blue-100 rounded-lg">
              <label className={labelCls}>从训练实验快速预填</label>
              <select className={inputCls + ' bg-white'}
                value={form.source_experiment_id}
                onChange={e => { set('source_experiment_id', e.target.value); fillFromExperiment(e.target.value) }}>
                <option value="">— 不使用 —</option>
                {experiments.filter(e => e.status === 'completed').map(e => (
                  <option key={e.id} value={e.id}>
                    #{e.id} {e.name} {e.config?.use_lora ? '(LoRA)' : ''}
                  </option>
                ))}
              </select>
              <p className="text-xs text-slate-500 mt-1">
                选中后会自动填入 model_path / lora_adapter_path / max_model_len
              </p>
            </div>
          )}

          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className={labelCls}>助手名称 *</label>
              <input className={inputCls} value={form.name} onChange={e => set('name', e.target.value)}
                placeholder="例：Qwen3-Coder-30B / GPT-4o-mini" />
            </div>
            <div>
              <label className={labelCls}>
                Model name *
                {form.type === 'local' && <span className="ml-1 text-slate-400">(即 vllm 的 served-model-name)</span>}
              </label>
              <input className={inputCls} value={form.model_name} onChange={e => set('model_name', e.target.value)}
                placeholder={form.type === 'local' ? 'Qwen3-Coder-30B-Instruct' : 'gpt-4o-mini'} />
            </div>
          </div>

          <div>
            <label className={labelCls}>描述（可选）</label>
            <input className={inputCls} value={form.description} onChange={e => set('description', e.target.value)}
              placeholder="对该助手的用途、备注等" />
          </div>

          {form.type === 'remote' ? (
            <>
              <div>
                <label className={labelCls}>Base URL *</label>
                <input className={inputCls} value={form.base_url} onChange={e => set('base_url', e.target.value)}
                  placeholder="https://api.openai.com/v1" />
              </div>
              <div>
                <label className={labelCls}>API Key {/^https?:\/\/(localhost|127\.0\.0\.1)/.test(form.base_url) && <span className="text-slate-400">(本地端点可空)</span>}</label>
                <div className="relative">
                  <input type={showKey ? 'text' : 'password'} className={inputCls + ' pr-16'} value={form.api_key}
                    onChange={e => set('api_key', e.target.value)}
                    placeholder={isEdit ? '留空则保持原 key 不变' : 'sk-...'} autoComplete="off" />
                  <button type="button" onClick={() => setShowKey(v => !v)}
                    className="absolute right-2 top-1/2 -translate-y-1/2 text-xs text-slate-400 hover:text-slate-600">
                    {showKey ? '隐藏' : '显示'}
                  </button>
                </div>
              </div>
            </>
          ) : (
            <>
              <div>
                <label className={labelCls}>Model path *</label>
                <input className={inputCls} value={form.model_path} onChange={e => set('model_path', e.target.value)}
                  placeholder="/home/xiaoyi/zkk_project/models/Qwen3-Coder-30B-A3B-Instruct-Int4-W4A16" />
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className={labelCls}>--max-model-len</label>
                  <input type="number" className={inputCls}
                    value={form.max_model_len || ''} onChange={e => set('max_model_len', e.target.value)}
                    placeholder="16384" />
                </div>
                <div>
                  <label className={labelCls}>LoRA adapter path（可选）</label>
                  <input className={inputCls} value={form.lora_adapter_path}
                    onChange={e => set('lora_adapter_path', e.target.value)}
                    placeholder="为 LoRA 微调结果时填入 adapter 目录" />
                </div>
              </div>

              {/* ── GPU selection ──────────────────────────────────────── */}
              <div className="border-t border-slate-100 pt-3">
                <div className="flex items-center justify-between mb-2">
                  <p className="text-xs font-medium text-slate-600">GPU 资源</p>
                  <button type="button" onClick={refreshGpus}
                    className="flex items-center gap-1 text-xs text-slate-500 hover:text-blue-600">
                    <RefreshCw size={11} className={gpuLoading ? 'animate-spin' : ''} /> 刷新
                  </button>
                </div>
                <div className="grid grid-cols-3 gap-2 mb-2">
                  {[
                    ['auto',   '自动',     '不修改 CUDA_VISIBLE_DEVICES'],
                    ['single', '单卡',     '指定一张 GPU 推理'],
                    ['multi',  '多卡 (TP)', 'tensor-parallel 多卡推理'],
                  ].map(([k, label, desc]) => (
                    <label key={k}
                      className={`flex items-start gap-1.5 p-2 rounded-lg border cursor-pointer transition-colors
                        ${gpuMode === k ? 'border-blue-400 bg-blue-50' : 'border-slate-200 hover:bg-slate-50'}`}>
                      <input type="radio" name="agpuMode" className="mt-0.5 accent-blue-600"
                        checked={gpuMode === k} onChange={() => setGpuMode(k)} />
                      <div className="min-w-0">
                        <p className="text-xs font-medium text-slate-700">{label}</p>
                        <p className="text-[10px] text-slate-400 mt-0.5">{desc}</p>
                      </div>
                    </label>
                  ))}
                </div>

                {(gpuMode === 'single' || gpuMode === 'multi') && (
                  availableGpus.length > 0 ? (
                    <div className="space-y-1.5">
                      {availableGpus.map(g => {
                        const isPicked = gpuMode === 'single'
                          ? pickedSingle === g.index
                          : pickedMulti.includes(g.index)
                        const usedPct = Math.round(g.memory_used_mb / g.memory_total_mb * 100)
                        return (
                          <label key={g.index}
                            className={`flex items-center gap-2.5 p-2 rounded-lg border cursor-pointer transition-colors text-xs
                              ${isPicked ? 'border-blue-400 bg-blue-50' : 'border-slate-200 hover:bg-slate-50'}`}>
                            <input
                              type={gpuMode === 'single' ? 'radio' : 'checkbox'}
                              name="agpuPick"
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
                            <div className="flex-1 min-w-0">
                              <p className="text-slate-700">
                                <span className="font-medium">GPU {g.index}</span>
                                <span className="ml-2 text-slate-500">{g.name}</span>
                              </p>
                              <p className="text-[10px] text-slate-400 mt-0.5">
                                显存 {g.memory_used_mb}/{g.memory_total_mb} MB ({usedPct}%) ·
                                空闲 {g.memory_free_mb} MB · 利用率 {g.utilization_pct}% · {g.temperature_c}°C
                              </p>
                            </div>
                          </label>
                        )
                      })}
                      {gpuMode === 'multi' && pickedMulti.length === 1 && (
                        <p className="text-xs text-amber-600">至少选择 2 张才会启用 tensor-parallel；当前只选 1 张将退化为单卡。</p>
                      )}
                    </div>
                  ) : (
                    <div className="space-y-1">
                      <p className="text-xs text-amber-600">
                        当前无法读取 GPU 列表（可能在开发机上）。可手动填写 GPU 索引。
                      </p>
                      <input type="text" value={manualIds}
                        onChange={e => setManualIds(e.target.value)}
                        placeholder={gpuMode === 'single' ? '例：0' : '例：0,1,2,3'}
                        className={inputCls} />
                    </div>
                  )
                )}

                <p className="text-[11px] text-slate-400 mt-2">
                  最终下发：
                  <code className="ml-1 px-1.5 py-0.5 bg-slate-100 rounded text-slate-700">
                    gpu_ids = {(() => {
                      const v = resolveGpuIds()
                      if (v === null) return 'null （auto）'
                      return `[${v.join(', ')}]`
                    })()}
                  </code>
                  {gpuMode === 'multi' && pickedMulti.length > 1 && (
                    <span className="ml-2">vllm 将自动追加 <code className="px-1 py-0.5 bg-slate-100 rounded">--tensor-parallel-size {pickedMulti.length}</code></span>
                  )}
                </p>
              </div>

              <div>
                <label className={labelCls}>额外 vllm 参数（空格分隔）</label>
                <input className={inputCls} value={form.extra_vllm_args}
                  onChange={e => set('extra_vllm_args', e.target.value)}
                  placeholder="例：--gpu-memory-utilization 0.85 --dtype bfloat16 --disable-log-requests" />
                <p className="text-[11px] text-slate-400 mt-1">
                  后端只附加 model / served-model-name / host / port / max-model-len / LoRA / tensor-parallel-size，其他参数完全由你掌控
                </p>
              </div>
            </>
          )}

          {err && <p className="text-xs text-red-500">{err}</p>}
        </div>

        <div className="px-6 py-4 border-t border-slate-100 flex justify-end gap-2">
          <button onClick={onClose}
            className="px-4 py-2 text-sm border border-slate-200 rounded-lg hover:bg-slate-50">取消</button>
          <button onClick={handleSubmit} disabled={loading}
            className="px-4 py-2 text-sm bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50">
            {loading ? '保存中...' : (isEdit ? '保存修改' : '创建助手')}
          </button>
        </div>
      </div>
    </div>
  )
}

// ─────────────────────────────── Assistant Card ────────────────────────────────────

function AssistantLogPanel({ assistantId }) {
  const [lines, setLines] = useState([])
  useEffect(() => {
    const fetchLog = () => assistantsApi.log(assistantId, 200)
      .then(r => setLines(r.data.lines || [])).catch(() => {})
    fetchLog()
    const t = setInterval(fetchLog, 3000)
    return () => clearInterval(t)
  }, [assistantId])
  return (
    <div className="mt-3 bg-slate-900 rounded-lg p-3 max-h-64 overflow-y-auto font-mono text-[11px] text-slate-200 leading-relaxed">
      {lines.length ? lines.map((l, i) => <div key={i} className="whitespace-pre-wrap">{l}</div>)
        : <span className="text-slate-500">暂无日志</span>}
    </div>
  )
}

function AssistantCard({ a, onChanged, onEdit }) {
  const [expanded, setExpanded] = useState(false)
  const [acting, setActing] = useState(false)
  const cfg = STATUS_CFG[a.status] || STATUS_CFG.stopped
  const Icon = cfg.icon

  const handleStart = async () => {
    setActing(true)
    try { await assistantsApi.start(a.id); onChanged() }
    catch (e) { alert(e?.response?.data?.detail || '启动失败') }
    setActing(false)
  }
  const handleStop = async () => {
    if (!confirm(`停止助手「${a.name}」？`)) return
    setActing(true)
    try { await assistantsApi.stop(a.id); onChanged() } catch {}
    setActing(false)
  }
  const handleDelete = async () => {
    if (!confirm(`删除助手「${a.name}」？此操作不可撤销。`)) return
    try { await assistantsApi.delete(a.id); onChanged() }
    catch (e) { alert(e?.response?.data?.detail || '删除失败') }
  }

  return (
    <div className="bg-white rounded-xl shadow-sm border border-slate-100 p-4">
      <div className="flex items-start justify-between gap-3">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            {a.type === 'local' ? <Server size={14} className="text-slate-500" /> : <Cloud size={14} className="text-slate-500" />}
            <span className="font-semibold text-slate-800 truncate">{a.name}</span>
            <span className={`flex items-center gap-1 px-2 py-0.5 rounded text-xs ${cfg.cls}`}>
              <Icon size={10} className={a.status === 'starting' ? 'animate-spin' : ''} /> {cfg.label}
            </span>
            <span className="px-2 py-0.5 rounded text-xs bg-slate-100 text-slate-600">
              {a.type === 'local' ? '本地 (vllm)' : '远程 API'}
            </span>
          </div>
          {a.description && <p className="text-xs text-slate-500 mt-1">{a.description}</p>}
          <div className="mt-1.5 text-xs text-slate-400 flex flex-wrap gap-x-4 gap-y-0.5">
            <span>model_name: <code className="text-slate-600">{a.model_name}</code></span>
            {a.type === 'local' && a.model_path && (
              <span title={a.model_path}>路径: <code className="text-slate-600">{a.model_path.length > 40 ? '…' + a.model_path.slice(-40) : a.model_path}</code></span>
            )}
            {a.type === 'local' && a.lora_adapter_path && <span className="text-purple-600">LoRA</span>}
            {a.type === 'local' && (
              <span className="text-blue-600">
                {!Array.isArray(a.gpu_ids) || a.gpu_ids.length === 0
                  ? 'GPU 自动'
                  : a.gpu_ids.length === 1
                    ? `GPU ${a.gpu_ids[0]}`
                    : `多卡 GPU [${a.gpu_ids.join(',')}] (TP=${a.gpu_ids.length})`}
              </span>
            )}
            {a.type === 'remote' && a.base_url && (
              <span title={a.base_url}>endpoint: <code className="text-slate-600">{a.base_url.slice(0, 32)}…</code></span>
            )}
            {a.port && <span>:{a.port}</span>}
            {a.has_api_key && <span className="text-green-600">✓ key</span>}
          </div>
          {a.error_message && (
            <p className="mt-1.5 text-xs text-red-500 break-all">⚠ {a.error_message.slice(0, 200)}</p>
          )}
        </div>

        <div className="flex items-center gap-1.5 shrink-0">
          {a.type === 'local' && (a.status === 'stopped' || a.status === 'failed') && (
            <button onClick={handleStart} disabled={acting}
              className="flex items-center gap-1 px-3 py-1.5 text-xs bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50">
              {acting ? <Loader2 size={12} className="animate-spin" /> : <Play size={12} />} 启动
            </button>
          )}
          {a.type === 'local' && (a.status === 'running' || a.status === 'starting') && (
            <button onClick={handleStop} disabled={acting}
              className="flex items-center gap-1 px-3 py-1.5 text-xs bg-red-500 text-white rounded-lg hover:bg-red-600 disabled:opacity-50">
              {acting ? <Loader2 size={12} className="animate-spin" /> : <Square size={12} />} 停止
            </button>
          )}
          {(a.type === 'local') && (
            <button onClick={() => setExpanded(v => !v)}
              className="flex items-center gap-1 px-3 py-1.5 text-xs border border-slate-200 rounded-lg hover:bg-slate-50">
              {expanded ? <ChevronUp size={12} /> : <ChevronDown size={12} />} 日志
            </button>
          )}
          <button onClick={onEdit}
            className="p-1.5 text-slate-400 hover:text-blue-600 border border-slate-200 rounded-lg hover:bg-blue-50">
            <Edit3 size={13} />
          </button>
          <button onClick={handleDelete}
            className="p-1.5 text-slate-400 hover:text-red-500 border border-slate-200 rounded-lg hover:bg-red-50">
            <Trash2 size={13} />
          </button>
        </div>
      </div>
      {expanded && a.type === 'local' && <AssistantLogPanel assistantId={a.id} />}
    </div>
  )
}

// ─────────────────────────────── Manage Tab ────────────────────────────────────────

function ManageTab() {
  const [list, setList] = useState([])
  const [exps, setExps] = useState([])
  const [loading, setLoading] = useState(false)
  const [showForm, setShowForm] = useState(false)
  const [editing, setEditing] = useState(null)

  const load = useCallback(() => {
    setLoading(true)
    assistantsApi.list().then(r => setList(r.data)).catch(() => {}).finally(() => setLoading(false))
  }, [])

  useEffect(() => {
    load()
    training.listExperiments().then(r => setExps(r.data)).catch(() => {})
  }, [load])

  // poll while any local assistant is starting
  useEffect(() => {
    if (list.some(a => a.status === 'starting')) {
      const t = setInterval(load, 3000)
      return () => clearInterval(t)
    }
  }, [list, load])

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <p className="text-sm text-slate-500">配置可在抽取 / 评估 / 演示中复用的 LLM 助手</p>
        <div className="flex gap-2">
          <button onClick={load}
            className="flex items-center gap-1.5 px-3 py-2 text-sm border border-slate-200 rounded-lg hover:bg-slate-50">
            <RefreshCw size={14} className={loading ? 'animate-spin' : ''} /> 刷新
          </button>
          <button onClick={() => { setEditing(null); setShowForm(true) }}
            className="flex items-center gap-1.5 px-3 py-2 text-sm bg-blue-600 text-white rounded-lg hover:bg-blue-700">
            <Plus size={14} /> 新建助手
          </button>
        </div>
      </div>

      <div className="space-y-3">
        {list.map(a => (
          <AssistantCard key={a.id} a={a} onChanged={load}
            onEdit={() => { setEditing(a); setShowForm(true) }} />
        ))}
        {!loading && list.length === 0 && (
          <div className="text-center py-16 text-slate-400">
            <Server size={36} className="mx-auto mb-3 opacity-30" />
            <p>暂无助手，点击「新建助手」开始</p>
          </div>
        )}
      </div>

      {showForm && (
        <AssistantFormModal
          initial={editing}
          experiments={exps}
          onClose={() => { setShowForm(false); setEditing(null) }}
          onSaved={load}
        />
      )}
    </div>
  )
}

// ─────────────────────────────── Demo Tab ──────────────────────────────────────────

function DemoTab() {
  const [list, setList] = useState([])
  const [pickedId, setPickedId] = useState('')
  const [input, setInput] = useState('')
  const [result, setResult] = useState(null)
  const [similar, setSimilar] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    assistantsApi.list().then(r => {
      const ready = r.data.filter(a => a.status === 'running')
      setList(ready)
      if (ready.length && !pickedId) setPickedId(String(ready[0].id))
    }).catch(() => {})
  }, [])

  const handleDiagnose = async () => {
    if (!input.trim()) return
    setLoading(true); setError(''); setResult(null); setSimilar([])
    try {
      const payload = pickedId
        ? { case_description: input, assistant_id: Number(pickedId) }
        : { case_description: input }
      const [diagRes, simRes] = await Promise.all([
        assistant.diagnose(payload),
        assistant.similarCases(input),
      ])
      setResult(diagRes.data.result)
      setSimilar(simRes.data)
    } catch (e) {
      setError(e?.response?.data?.detail || '请求失败')
    }
    setLoading(false)
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
      <div className="lg:col-span-3 space-y-4">
        <div className="bg-white rounded-xl shadow-sm border border-slate-100 p-5">
          <label className="text-sm font-medium text-slate-700 mb-2 block">使用助手</label>
          <select value={pickedId} onChange={e => setPickedId(e.target.value)}
            className="w-full px-3 py-2 text-sm border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 mb-4">
            <option value="">— 使用 .env 默认配置 —</option>
            {list.map(a => (
              <option key={a.id} value={a.id}>
                {a.type === 'local' ? '🖥' : '☁'} {a.name} ({a.model_name})
              </option>
            ))}
          </select>
          {list.length === 0 && (
            <p className="mb-3 p-2 bg-amber-50 text-amber-700 text-xs rounded">
              当前无运行中的助手。可在「助手管理」标签页配置并启动一个本地或远程助手。
            </p>
          )}

          <label className="text-sm font-medium text-slate-700 mb-2 block">病情描述</label>
          <textarea
            className="w-full h-44 px-3 py-2 text-sm border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none"
            placeholder="请输入患者主诉、现病史、体格检查、辅助检查等信息..."
            value={input}
            onChange={e => setInput(e.target.value)}
          />
          <button onClick={handleDiagnose} disabled={loading || !input.trim()}
            className="mt-3 w-full flex items-center justify-center gap-2 py-2.5 bg-blue-600 text-white text-sm rounded-lg hover:bg-blue-700 disabled:opacity-50">
            {loading ? <><Loader2 size={14} className="animate-spin" /> 分析中...</>
                     : <><Send size={14} /> 开始分析</>}
          </button>
          {error && <p className="mt-2 text-xs text-red-500">{error}</p>}
        </div>

        {result && (
          <div className="bg-white rounded-xl shadow-sm border border-slate-100 p-5">
            <div className="flex items-center gap-2 mb-3">
              <Stethoscope size={16} className="text-blue-600" />
              <h2 className="text-sm font-semibold text-slate-700">诊断分析结果</h2>
            </div>
            {result.raw
              ? <MarkdownView source={result.raw} />
              : (
                <pre className="text-sm text-slate-700 whitespace-pre-wrap font-sans leading-relaxed">
                  {JSON.stringify(result, null, 2)}
                </pre>
              )}
          </div>
        )}
      </div>

      <div className="lg:col-span-2">
        <div className="bg-white rounded-xl shadow-sm border border-slate-100 p-5 sticky top-6">
          <h2 className="text-sm font-semibold text-slate-700 mb-3">相似病例</h2>
          {similar.length > 0 ? (
            <div className="space-y-2">
              {similar.map(c => (
                <div key={c.id} className="p-3 bg-slate-50 rounded-lg">
                  <div className="flex items-start justify-between gap-2">
                    <p className="text-xs text-slate-700 font-medium leading-snug flex-1 min-w-0">{c.title}</p>
                    {c.score != null && (
                      <span className="px-1.5 py-0.5 bg-blue-100 text-blue-700 rounded text-[10px] shrink-0">
                        匹配 {c.score}
                      </span>
                    )}
                  </div>
                  {c.matched_keywords?.length > 0 && (
                    <div className="mt-1 flex flex-wrap gap-1">
                      {c.matched_keywords.map(kw => (
                        <span key={kw} className="px-1 py-0.5 bg-amber-50 text-amber-700 rounded text-[10px]">{kw}</span>
                      ))}
                    </div>
                  )}
                  {c.snippet && (
                    <p className="mt-1.5 text-[11px] text-slate-500 leading-relaxed line-clamp-3">{c.snippet}</p>
                  )}
                  <span className="block mt-1 text-[10px] text-slate-400">{c.type === 'case_report' ? '病例报告' : '指南'}</span>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-xs text-slate-400">输入病情描述后将显示相似病例</p>
          )}
        </div>
      </div>
    </div>
  )
}

// ─────────────────────────────── Page ──────────────────────────────────────────────

export default function Assistant() {
  const [tab, setTab] = useState('demo')

  return (
    <div className="p-8 max-w-6xl mx-auto">
      <div className="mb-6">
        <h1 className="text-xl font-bold text-slate-800">临床助手</h1>
        <p className="text-sm text-slate-500 mt-0.5">管理可复用的 LLM 助手；体验诊断分析功能</p>
      </div>

      <div className="border-b border-slate-200 mb-5">
        <div className="flex gap-1">
          {[['demo', '临床演示'], ['manage', '助手管理']].map(([k, label]) => (
            <button key={k} onClick={() => setTab(k)}
              className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors -mb-px
                ${tab === k ? 'border-blue-600 text-blue-700' : 'border-transparent text-slate-500 hover:text-slate-700'}`}>
              {label}
            </button>
          ))}
        </div>
      </div>

      {tab === 'demo' && <DemoTab />}
      {tab === 'manage' && <ManageTab />}
    </div>
  )
}
