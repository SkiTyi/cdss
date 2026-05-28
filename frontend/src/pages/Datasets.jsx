import { useEffect, useState } from 'react'
import { datasets, extraction } from '../api/client'
import { Plus, Download, Trash2, Scissors, BarChart2 } from 'lucide-react'

const STRATEGY_LABEL = {
  case_direct:      'base · 病例直出',
  guideline_synth:  'base · 指南合成',
  aug_paraphrase:   'aug · 改写',
  aug_distractor:   'aug · 干扰',
  aug_cot:          'aug · CoT 扩写',
  aug_hardneg:      'aug · 困难负样本',
  aug_comorbidity:  'aug · 合并症',
}

function CreateDatasetModal({ onClose, onCreated }) {
  const [form, setForm] = useState({
    name: '',
    description: '',
    format: 'alpaca',
    job_id: '',
    approved_only: false,
    include_strategies: [],          // empty = include all
    sampling_strategy: 'proportional',
    max_per_disease: '',
    seed: '',
    system_prompt: '你是一位专业的临床医学助手，请根据患者的病情描述给出专业的诊断分析和治疗建议。',
  })
  const [jobs, setJobs] = useState([])
  const [preview, setPreview] = useState(null)      // {total_candidates, projected_count, ...}
  const [previewLoading, setPreviewLoading] = useState(false)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    extraction.listJobs().then(r => setJobs(r.data.filter(j => j.status === 'completed')))
  }, [])

  // Auto-refresh preview when any filter/sampling param changes (debounced)
  useEffect(() => {
    const filtersReady = form.job_id || form.include_strategies.length > 0
    if (!filtersReady) { setPreview(null); return }
    const t = setTimeout(async () => {
      setPreviewLoading(true)
      try {
        const r = await datasets.previewSource({
          job_id: form.job_id ? Number(form.job_id) : undefined,
          approved_only: form.approved_only,
          include_strategies: form.include_strategies.length ? form.include_strategies : undefined,
          sampling_strategy: form.sampling_strategy,
          max_per_disease: form.max_per_disease ? Number(form.max_per_disease) : undefined,
          top: 15,
        })
        setPreview(r.data)
      } catch {
        setPreview(null)
      }
      setPreviewLoading(false)
    }, 350)
    return () => clearTimeout(t)
  }, [form.job_id, form.approved_only, form.include_strategies.join(','),
      form.sampling_strategy, form.max_per_disease])

  const availableStrategies = preview?.strategy_breakdown
    ? Object.keys(preview.strategy_breakdown)
    : []

  const toggleStrategy = (s) => {
    setForm(f => {
      const set = new Set(f.include_strategies)
      set.has(s) ? set.delete(s) : set.add(s)
      return { ...f, include_strategies: [...set] }
    })
  }

  const handleSubmit = async () => {
    if (!form.name) return
    setLoading(true)
    try {
      await datasets.create({
        name: form.name,
        description: form.description || undefined,
        format: form.format,
        job_id: form.job_id ? Number(form.job_id) : undefined,
        approved_only: form.approved_only,
        include_strategies: form.include_strategies.length ? form.include_strategies : undefined,
        sampling_strategy: form.sampling_strategy,
        max_per_disease: form.max_per_disease ? Number(form.max_per_disease) : undefined,
        seed: form.seed ? Number(form.seed) : undefined,
        system_prompt: form.system_prompt,
      })
      onCreated()
      onClose()
    } catch {}
    setLoading(false)
  }

  const inputCls = 'w-full px-3 py-2 text-sm border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500'

  const headBarMax = preview?.head?.[0]?.count || 1

  return (
    <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-xl shadow-xl w-full max-w-4xl max-h-[92vh] overflow-hidden flex">

        {/* 左侧：表单 */}
        <div className="w-1/2 p-6 overflow-y-auto border-r border-slate-100">
          <h2 className="text-base font-semibold mb-4">新建数据集</h2>

          <div className="space-y-3">
            <div>
              <label className="text-xs text-slate-500 mb-1 block">数据集名称 *</label>
              <input className={inputCls} value={form.name}
                onChange={e => setForm(f => ({ ...f, name: e.target.value }))}
                placeholder="例：医学QA数据集-v1" />
            </div>

            <div>
              <label className="text-xs text-slate-500 mb-1 block">描述（可选）</label>
              <input className={inputCls} value={form.description}
                onChange={e => setForm(f => ({ ...f, description: e.target.value }))} />
            </div>

            <div>
              <label className="text-xs text-slate-500 mb-1 block">来源抽取任务</label>
              <select className={inputCls}
                value={form.job_id} onChange={e => setForm(f => ({ ...f, job_id: e.target.value }))}>
                <option value="">全部已完成任务</option>
                {jobs.map(j => (
                  <option key={j.id} value={j.id}>
                    #{j.id} {j.name} ({j.task_type})
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="flex items-center gap-2 text-xs text-slate-600 cursor-pointer">
                <input type="checkbox" className="accent-blue-600"
                  checked={form.approved_only}
                  onChange={e => setForm(f => ({ ...f, approved_only: e.target.checked }))} />
                只用已审核条目
              </label>
            </div>

            {availableStrategies.length > 0 && (
              <div>
                <label className="text-xs text-slate-500 mb-1 block">
                  合成策略筛选（不勾则全选）
                </label>
                <div className="flex flex-wrap gap-1">
                  {availableStrategies.map(s => {
                    const active = form.include_strategies.includes(s)
                    const n = preview.strategy_breakdown[s]
                    return (
                      <button key={s} type="button" onClick={() => toggleStrategy(s)}
                        className={`px-2 py-1 text-[11px] rounded border transition-colors
                          ${active ? 'bg-blue-100 text-blue-700 border-blue-300' : 'bg-slate-50 text-slate-600 border-slate-200 hover:bg-slate-100'}`}>
                        {STRATEGY_LABEL[s] || s} ({n})
                      </button>
                    )
                  })}
                </div>
              </div>
            )}

            <div className="border-t border-slate-100 pt-3">
              <p className="text-xs font-medium text-slate-600 mb-2">采样策略</p>
              <div className="space-y-1.5">
                {[
                  ['proportional', '按比例（natural distribution）', '保持原始诊断分布，可选 max_per_disease 截断长尾头部'],
                  ['uniform_by_disease', '按诊断均匀（uniform）', '每个诊断取相同条数（防止头部疾病淹没尾部）'],
                  ['none', '不采样（原样使用）', '所有候选全用'],
                ].map(([k, label, desc]) => (
                  <label key={k}
                    className={`flex items-start gap-2 p-2 rounded border cursor-pointer text-xs
                      ${form.sampling_strategy === k ? 'border-blue-400 bg-blue-50' : 'border-slate-200 hover:bg-slate-50'}`}>
                    <input type="radio" name="sampling" className="mt-0.5 accent-blue-600"
                      checked={form.sampling_strategy === k}
                      onChange={() => setForm(f => ({ ...f, sampling_strategy: k }))} />
                    <div className="min-w-0 flex-1">
                      <p className="text-slate-700 font-medium">{label}</p>
                      <p className="text-[11px] text-slate-400">{desc}</p>
                    </div>
                  </label>
                ))}
              </div>
              {form.sampling_strategy !== 'none' && (
                <div className="mt-2 grid grid-cols-2 gap-2">
                  <div>
                    <label className="text-xs text-slate-500 mb-0.5 block">
                      max_per_disease {form.sampling_strategy === 'uniform_by_disease' && '*'}
                    </label>
                    <input type="number" min="1" step="1" className={inputCls}
                      value={form.max_per_disease}
                      onChange={e => setForm(f => ({ ...f, max_per_disease: e.target.value }))}
                      placeholder={form.sampling_strategy === 'uniform_by_disease' ? '默认 = 最小桶大小' : '留空=不截断'} />
                  </div>
                  <div>
                    <label className="text-xs text-slate-500 mb-0.5 block">随机种子（可选）</label>
                    <input type="number" className={inputCls} value={form.seed}
                      onChange={e => setForm(f => ({ ...f, seed: e.target.value }))}
                      placeholder="留空=不可复现" />
                  </div>
                </div>
              )}
            </div>

            <div className="border-t border-slate-100 pt-3">
              <label className="text-xs text-slate-500 mb-1 block">数据格式</label>
              <select className={inputCls}
                value={form.format} onChange={e => setForm(f => ({ ...f, format: e.target.value }))}>
                <option value="alpaca">Alpaca（instruction/input/output）</option>
                <option value="sharegpt">ShareGPT（conversations）</option>
              </select>
            </div>

            <div>
              <label className="text-xs text-slate-500 mb-1 block">系统提示词</label>
              <textarea className={inputCls + ' h-16 resize-none'}
                value={form.system_prompt}
                onChange={e => setForm(f => ({ ...f, system_prompt: e.target.value }))} />
            </div>
          </div>

          <div className="flex justify-end gap-2 mt-5">
            <button onClick={onClose}
              className="px-4 py-2 text-sm border border-slate-200 rounded-lg hover:bg-slate-50">取消</button>
            <button onClick={handleSubmit} disabled={loading || !form.name}
              className="px-4 py-2 text-sm bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50">
              {loading ? '构建中...' : `构建 (${preview?.projected_count ?? '?'} 条)`}
            </button>
          </div>
        </div>

        {/* 右侧：预览 */}
        <div className="w-1/2 p-6 overflow-y-auto bg-slate-50/50">
          <div className="flex items-center gap-2 mb-4">
            <BarChart2 size={16} className="text-slate-500" />
            <h3 className="text-sm font-semibold text-slate-700">采样预览</h3>
            {previewLoading && <span className="text-[11px] text-slate-400">计算中...</span>}
          </div>

          {!preview && (
            <div className="text-center text-xs text-slate-400 py-12">
              选择来源任务后将自动计算分布与采样后条数
            </div>
          )}

          {preview && (
            <>
              <div className="grid grid-cols-2 gap-2 mb-4">
                <div className="bg-white p-3 rounded-lg border border-slate-100">
                  <p className="text-[11px] text-slate-500">候选总数</p>
                  <p className="text-lg font-semibold text-slate-700">{preview.total_candidates.toLocaleString()}</p>
                </div>
                <div className="bg-white p-3 rounded-lg border border-blue-200">
                  <p className="text-[11px] text-blue-500">采样后</p>
                  <p className="text-lg font-semibold text-blue-700">{preview.projected_count.toLocaleString()}</p>
                </div>
                <div className="bg-white p-3 rounded-lg border border-slate-100">
                  <p className="text-[11px] text-slate-500">诊断种数</p>
                  <p className="text-base font-semibold text-slate-700">{preview.distinct_diagnoses}</p>
                </div>
                <div className="bg-white p-3 rounded-lg border border-slate-100">
                  <p className="text-[11px] text-slate-500">单例诊断（仅 1 条）</p>
                  <p className="text-base font-semibold text-amber-600">{preview.singletons}</p>
                </div>
              </div>

              <div className="mb-4">
                <p className="text-[11px] text-slate-500 mb-1.5">合成策略分布</p>
                <div className="space-y-1">
                  {Object.entries(preview.strategy_breakdown).map(([s, n]) => (
                    <div key={s} className="flex items-center justify-between text-[11px]">
                      <span className="text-slate-600">{STRATEGY_LABEL[s] || s}</span>
                      <span className="text-slate-700 font-mono">{n}</span>
                    </div>
                  ))}
                </div>
              </div>

              <div>
                <p className="text-[11px] text-slate-500 mb-1.5">诊断分布（前 {preview.head.length}）</p>
                <div className="space-y-1">
                  {preview.head.map(h => {
                    const candPct = (h.count / headBarMax) * 100
                    const projPct = (h.projected_count / headBarMax) * 100
                    return (
                      <div key={h.label} className="text-[11px]">
                        <div className="flex items-center justify-between mb-0.5">
                          <span className="text-slate-700 truncate max-w-[60%]" title={h.label}>{h.label || '(未标注)'}</span>
                          <span className="text-slate-500 font-mono">
                            {h.count}
                            {h.projected_count !== h.count && (
                              <span className="text-blue-600"> → {h.projected_count}</span>
                            )}
                          </span>
                        </div>
                        <div className="relative h-2 bg-slate-100 rounded overflow-hidden">
                          <div className="absolute inset-y-0 left-0 bg-slate-300" style={{ width: `${candPct}%` }} />
                          <div className="absolute inset-y-0 left-0 bg-blue-500" style={{ width: `${projPct}%` }} />
                        </div>
                      </div>
                    )
                  })}
                </div>
              </div>
            </>
          )}
        </div>

      </div>
    </div>
  )
}

function SplitDatasetModal({ src, onClose, onCreated }) {
  const [form, setForm] = useState({
    name: `${src.name}-split`,
    mode: 'sample_size',           // sample_size | ratio
    sample_size: Math.min(50, src.item_count),
    ratio: 0.1,
    seed: '',
  })
  const [loading, setLoading] = useState(false)
  const [err, setErr] = useState('')

  const handleSubmit = async () => {
    setErr('')
    if (!form.name.trim()) return
    setLoading(true)
    try {
      const payload = {
        name: form.name.trim(),
        description: `从 #${src.id} (${src.name}) 随机切分`,
      }
      if (form.mode === 'sample_size') payload.sample_size = Number(form.sample_size)
      else payload.ratio = Number(form.ratio)
      if (form.seed) payload.seed = Number(form.seed)
      await datasets.split(src.id, payload)
      onCreated()
      onClose()
    } catch (e) {
      setErr(e?.response?.data?.detail || '切分失败')
    }
    setLoading(false)
  }

  const inputCls = 'w-full px-3 py-2 text-sm border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500'

  return (
    <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50">
      <div className="bg-white rounded-xl shadow-xl w-full max-w-md p-6">
        <h2 className="text-base font-semibold mb-1">随机切分数据集</h2>
        <p className="text-xs text-slate-500 mb-4">从「{src.name}」({src.item_count} 条) 随机抽取一个子集，作为新数据集</p>
        <div className="space-y-3">
          <div>
            <label className="text-xs text-slate-500 mb-1 block">新数据集名称 *</label>
            <input className={inputCls} value={form.name} onChange={e => setForm(f => ({ ...f, name: e.target.value }))} />
          </div>

          <div className="grid grid-cols-2 gap-2">
            {[['sample_size', '按数量'], ['ratio', '按比例']].map(([k, label]) => (
              <label key={k} className={`flex items-center gap-2 p-2.5 rounded-lg border cursor-pointer
                ${form.mode === k ? 'border-blue-400 bg-blue-50' : 'border-slate-200'}`}>
                <input type="radio" name="mode" className="accent-blue-600"
                  checked={form.mode === k} onChange={() => setForm(f => ({ ...f, mode: k }))} />
                <span className="text-sm text-slate-700">{label}</span>
              </label>
            ))}
          </div>

          {form.mode === 'sample_size' ? (
            <div>
              <label className="text-xs text-slate-500 mb-1 block">抽样数量（最多 {src.item_count}）</label>
              <input type="number" min="1" max={src.item_count}
                className={inputCls} value={form.sample_size}
                onChange={e => setForm(f => ({ ...f, sample_size: e.target.value }))} />
            </div>
          ) : (
            <div>
              <label className="text-xs text-slate-500 mb-1 block">
                抽样比例：{(form.ratio * 100).toFixed(0)}% （约 {Math.max(1, Math.round(src.item_count * form.ratio))} 条）
              </label>
              <input type="range" min="0.01" max="1" step="0.01"
                className="w-full accent-blue-600" value={form.ratio}
                onChange={e => setForm(f => ({ ...f, ratio: parseFloat(e.target.value) }))} />
            </div>
          )}

          <div>
            <label className="text-xs text-slate-500 mb-1 block">随机种子（可选，便于复现）</label>
            <input type="number" className={inputCls} value={form.seed}
              onChange={e => setForm(f => ({ ...f, seed: e.target.value }))} placeholder="留空则每次切分都不同" />
          </div>
        </div>
        {err && <p className="mt-3 text-xs text-red-500">{err}</p>}
        <div className="flex justify-end gap-2 mt-5">
          <button onClick={onClose}
            className="px-4 py-2 text-sm border border-slate-200 rounded-lg hover:bg-slate-50">取消</button>
          <button onClick={handleSubmit} disabled={loading || !form.name.trim()}
            className="px-4 py-2 text-sm bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50">
            {loading ? '切分中...' : '生成新数据集'}
          </button>
        </div>
      </div>
    </div>
  )
}

export default function Datasets() {
  const [list, setList] = useState([])
  const [showCreate, setShowCreate] = useState(false)
  const [splitting, setSplitting] = useState(null)
  const [preview, setPreview] = useState(null)
  const [previewItems, setPreviewItems] = useState([])

  const load = () => datasets.list().then(r => setList(r.data))

  useEffect(() => { load() }, [])

  const handlePreview = async (ds) => {
    setPreview(ds)
    const r = await datasets.items(ds.id, { page: 1, page_size: 5 })
    setPreviewItems(r.data.items)
  }

  const handleDelete = async (id) => {
    if (!confirm('确认删除该数据集？')) return
    await datasets.delete(id)
    load()
  }

  return (
    <div className="p-8">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-xl font-bold text-slate-800">数据集管理</h1>
          <p className="text-sm text-slate-500 mt-0.5">从抽取的知识条目构建训练数据集</p>
        </div>
        <button onClick={() => setShowCreate(true)} className="flex items-center gap-1.5 px-3 py-2 text-sm bg-blue-600 text-white rounded-lg hover:bg-blue-700">
          <Plus size={14} /> 新建数据集
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {list.map(ds => (
          <div key={ds.id} className="bg-white rounded-xl shadow-sm border border-slate-100 p-5">
            <div className="flex items-start justify-between mb-2">
              <div>
                <h3 className="font-medium text-slate-800">{ds.name}</h3>
                <p className="text-xs text-slate-400 mt-0.5">{ds.description || '无描述'}</p>
              </div>
              <span className={`px-2 py-0.5 rounded text-xs ${ds.status === 'ready' ? 'bg-green-100 text-green-700' : 'bg-slate-100 text-slate-600'}`}>
                {ds.status === 'ready' ? '就绪' : '构建中'}
              </span>
            </div>
            <div className="flex items-center gap-4 text-xs text-slate-500 mb-4">
              <span>{ds.item_count} 条样本</span>
              <span>格式：{ds.format}</span>
              <span>{ds.created_at?.slice(0, 10)}</span>
            </div>
            <div className="flex gap-2">
              <button onClick={() => handlePreview(ds)} className="flex-1 py-1.5 text-xs border border-slate-200 rounded-lg hover:bg-slate-50">预览</button>
              <button onClick={() => setSplitting(ds)} title="随机切分"
                className="flex items-center justify-center gap-1 flex-1 py-1.5 text-xs border border-slate-200 rounded-lg hover:bg-slate-50 text-slate-700">
                <Scissors size={12} /> 切分
              </button>
              <a href={datasets.exportUrl(ds.id)} download className="flex items-center justify-center gap-1 flex-1 py-1.5 text-xs bg-blue-50 text-blue-700 rounded-lg hover:bg-blue-100">
                <Download size={12} /> 导出
              </a>
              <button onClick={() => handleDelete(ds.id)} className="p-1.5 text-slate-400 hover:text-red-500 border border-slate-200 rounded-lg hover:bg-red-50">
                <Trash2 size={14} />
              </button>
            </div>
          </div>
        ))}
        {list.length === 0 && (
          <div className="col-span-2 text-center py-12 text-slate-400">暂无数据集，请先完成知识抽取后构建</div>
        )}
      </div>

      {preview && (
        <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50">
          <div className="bg-white rounded-xl shadow-xl w-full max-w-2xl p-6 max-h-[80vh] overflow-y-auto">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-base font-semibold">{preview.name} — 预览（前5条）</h2>
              <button onClick={() => setPreview(null)} className="text-slate-400 hover:text-slate-600">✕</button>
            </div>
            <div className="space-y-3">
              {previewItems.map((item, i) => (
                <div key={i} className="p-3 bg-slate-50 rounded-lg text-xs space-y-1">
                  <div><span className="text-slate-400">instruction: </span><span className="text-slate-700">{item.instruction}</span></div>
                  {item.input && <div><span className="text-slate-400">input: </span><span className="text-slate-700">{item.input}</span></div>}
                  <div><span className="text-slate-400">output: </span><span className="text-slate-700">{item.output?.slice(0, 200)}</span></div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {showCreate && <CreateDatasetModal onClose={() => setShowCreate(false)} onCreated={load} />}
      {splitting && (
        <SplitDatasetModal src={splitting}
          onClose={() => setSplitting(null)}
          onCreated={load} />
      )}
    </div>
  )
}
