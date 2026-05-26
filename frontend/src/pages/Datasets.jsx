import { useEffect, useState } from 'react'
import { datasets, extraction } from '../api/client'
import { Plus, Download, Trash2, Scissors } from 'lucide-react'

function CreateDatasetModal({ onClose, onCreated }) {
  const [form, setForm] = useState({ name: '', description: '', format: 'alpaca', job_id: '', system_prompt: '你是一位专业的临床医学助手，请根据患者的病情描述给出专业的诊断分析和治疗建议。' })
  const [jobs, setJobs] = useState([])
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    extraction.listJobs().then(r => setJobs(r.data.filter(j => j.status === 'completed')))
  }, [])

  const handleSubmit = async () => {
    if (!form.name) return
    setLoading(true)
    try {
      await datasets.create({ ...form, job_id: form.job_id ? Number(form.job_id) : undefined })
      onCreated()
      onClose()
    } catch {}
    setLoading(false)
  }

  return (
    <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50">
      <div className="bg-white rounded-xl shadow-xl w-full max-w-md p-6">
        <h2 className="text-base font-semibold mb-4">新建数据集</h2>
        <div className="space-y-3">
          <div>
            <label className="text-xs text-slate-500 mb-1 block">数据集名称</label>
            <input className="w-full px-3 py-2 text-sm border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              value={form.name} onChange={e => setForm(f => ({ ...f, name: e.target.value }))} placeholder="例：医学QA数据集-v1" />
          </div>
          <div>
            <label className="text-xs text-slate-500 mb-1 block">来源抽取任务（留空使用全部QA条目）</label>
            <select className="w-full px-3 py-2 text-sm border border-slate-200 rounded-lg focus:outline-none"
              value={form.job_id} onChange={e => setForm(f => ({ ...f, job_id: e.target.value }))}>
              <option value="">全部已完成任务</option>
              {jobs.map(j => <option key={j.id} value={j.id}>{j.name}</option>)}
            </select>
          </div>
          <div>
            <label className="text-xs text-slate-500 mb-1 block">数据格式</label>
            <select className="w-full px-3 py-2 text-sm border border-slate-200 rounded-lg focus:outline-none"
              value={form.format} onChange={e => setForm(f => ({ ...f, format: e.target.value }))}>
              <option value="alpaca">Alpaca（instruction/input/output）</option>
              <option value="sharegpt">ShareGPT（conversations）</option>
            </select>
          </div>
          <div>
            <label className="text-xs text-slate-500 mb-1 block">系统提示词</label>
            <textarea className="w-full px-3 py-2 text-sm border border-slate-200 rounded-lg focus:outline-none h-16 resize-none"
              value={form.system_prompt} onChange={e => setForm(f => ({ ...f, system_prompt: e.target.value }))} />
          </div>
        </div>
        <div className="flex justify-end gap-2 mt-5">
          <button onClick={onClose} className="px-4 py-2 text-sm border border-slate-200 rounded-lg hover:bg-slate-50">取消</button>
          <button onClick={handleSubmit} disabled={loading || !form.name}
            className="px-4 py-2 text-sm bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50">
            {loading ? '构建中...' : '构建数据集'}
          </button>
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
