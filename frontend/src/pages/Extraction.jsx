import { useEffect, useState, useRef } from 'react'
import { extraction, assistants as assistantsApi, documents as documentsApi } from '../api/client'
import { Plus, RefreshCw, CheckCircle, XCircle, Clock, Play, Pause, Trash2, RotateCcw, ChevronDown, ChevronUp } from 'lucide-react'

const STATUS_CONFIG = {
  pending: { label: '待运行', color: 'bg-slate-100 text-slate-600', icon: Clock },
  running: { label: '运行中', color: 'bg-blue-100 text-blue-700', icon: Play },
  paused: { label: '已暂停', color: 'bg-yellow-100 text-yellow-700', icon: Pause },
  completed: { label: '已完成', color: 'bg-green-100 text-green-700', icon: CheckCircle },
  failed: { label: '失败', color: 'bg-red-100 text-red-700', icon: XCircle },
}

function CreateJobModal({ onClose, onCreated, existingJobs }) {
  const [form, setForm] = useState({
    name: '',
    task_type: 'case_extract',          // case_extract | guideline_synth | case_reasoning | augment
    llm_mode: 'assistant',              // assistant | manual
    assistant_id: '',
    base_url: '',
    model: '',
    api_key: '',
    prompt_template: '',
    doc_limit: '',
    // guideline_synth
    n_per_doc: 8,
    // augment
    source_job_id: '',
    augment_strategies: [],
    max_source_instances: 200,
    variants_per_strategy: 1,
  })
  const [defaults, setDefaults] = useState({})
  const [assistantList, setAssistantList] = useState([])
  const [docStats, setDocStats] = useState(null)
  const [augStrategies, setAugStrategies] = useState([])
  const [loading, setLoading] = useState(false)
  const [showKey, setShowKey] = useState(false)
  const [err, setErr] = useState('')

  useEffect(() => {
    extraction.defaultPrompts().then(r => setDefaults(r.data))
    documentsApi.stats().then(r => setDocStats(r.data)).catch(() => {})
    extraction.augmentStrategies().then(r => {
      setAugStrategies(r.data.strategies || [])
      setForm(f => ({ ...f, augment_strategies: r.data.defaults || [] }))
    }).catch(() => {})
    assistantsApi.list().then(r => {
      const ready = r.data.filter(a => a.status === 'running')
      setAssistantList(ready)
      if (ready.length && !form.assistant_id) {
        setForm(f => ({ ...f, assistant_id: String(ready[0].id) }))
      } else if (!ready.length) {
        setForm(f => ({ ...f, llm_mode: 'manual' }))
      }
    }).catch(() => setForm(f => ({ ...f, llm_mode: 'manual' })))
  }, [])

  const useAssistant = form.llm_mode === 'assistant'
  const isAugment = form.task_type === 'augment'
  const isGuidelineSynth = form.task_type === 'guideline_synth'

  // task_type implies which document type the job will pull from.
  const docTypeForTask = isGuidelineSynth ? 'guideline' : 'case_report'
  const availableDocs = (docStats == null || isAugment) ? null
    : docTypeForTask === 'guideline' ? docStats.guidelines : docStats.case_reports

  // Completed extraction jobs that can serve as source for augment
  const augmentableJobs = (existingJobs || []).filter(j =>
    j.status === 'completed' && ['case_extract', 'guideline_synth', 'case_reasoning'].includes(j.task_type)
  )

  const isLocal = /(localhost|127\.0\.0\.1|0\.0\.0\.0|::1)/i.test(form.base_url)
  const needKey = !useAssistant && form.base_url.trim() && !isLocal && !form.api_key.trim()

  const defaultTemplateKey = form.task_type

  const toggleStrategy = (name) => {
    setForm(f => {
      const set = new Set(f.augment_strategies || [])
      set.has(name) ? set.delete(name) : set.add(name)
      return { ...f, augment_strategies: [...set] }
    })
  }

  const handleSubmit = async () => {
    setErr('')
    if (!form.name) return
    if (useAssistant && !form.assistant_id) { setErr('请选择一个助手'); return }
    if (needKey) { setErr('远程 base_url 必须提供 api_key（仅 localhost 可为空）'); return }
    if (isAugment) {
      if (!form.source_job_id) { setErr('请选择源任务'); return }
      if (!form.augment_strategies?.length) { setErr('请至少勾选一项增强策略'); return }
    }
    setLoading(true)
    try {
      const payload = {
        name: form.name,
        task_type: form.task_type,
        prompt_template: form.prompt_template || undefined,
        doc_limit: form.doc_limit ? parseInt(form.doc_limit) : undefined,
        config: {},
      }
      if (useAssistant) {
        payload.assistant_id = Number(form.assistant_id)
      } else {
        payload.base_url = form.base_url.trim() || undefined
        payload.model = form.model.trim() || undefined
        payload.api_key = form.api_key.trim() || undefined
      }
      if (isGuidelineSynth) {
        payload.config.n_per_doc = Number(form.n_per_doc) || 8
      }
      if (isAugment) {
        payload.source_job_id = Number(form.source_job_id)
        payload.augment_strategies = form.augment_strategies
        payload.config.max_source_instances = Number(form.max_source_instances) || 200
        payload.config.variants_per_strategy = Number(form.variants_per_strategy) || 1
      }
      await extraction.createJob(payload)
      onCreated()
      onClose()
    } catch (e) {
      setErr(e?.response?.data?.detail || '创建失败')
    }
    setLoading(false)
  }

  return (
    <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50">
      <div className="bg-white rounded-xl shadow-xl w-full max-w-lg p-6 max-h-[90vh] overflow-y-auto">
        <h2 className="text-base font-semibold mb-4">新建抽取任务</h2>
        <div className="space-y-3">
          <div>
            <label className="text-xs text-slate-500 mb-1 block">任务名称 *</label>
            <input className="w-full px-3 py-2 text-sm border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              value={form.name} onChange={e => setForm(f => ({ ...f, name: e.target.value }))} placeholder="例：病例报告抽取测试-v1" />
          </div>

          {/* Task type selector */}
          <div>
            <label className="text-xs text-slate-500 mb-1 block">任务类型 *</label>
            <div className="space-y-2">
              {[
                ['case_extract', '病例 → 单实例', 'case_report',
                  '每篇病例产出 1 条 (presentation, answer) 训练样本：完整脱敏就诊场景 + 5 步推理 answer'],
                ['guideline_synth', '指南 → N 个虚拟患者', 'guideline',
                  '为指南诊断标准合成 N 个不同年龄/性别/合并症/严重度的虚拟患者就诊场景（产出密度提升一个数量级）'],
                ['case_reasoning', '病例多场景合成', 'case_report',
                  '每篇病例产出 2~4 条高多样性场景（诊断推理/鉴别/治疗规划）'],
                ['augment', '增强：基于已有 job 扩样本', null,
                  '对一个已完成 job 的 base instance 跑改写/干扰/CoT 扩写等策略，每条 base 产 N 个变体'],
              ].map(([k, label, srcType, desc]) => (
                <label key={k}
                  className={`flex items-start gap-2 p-3 rounded-lg border cursor-pointer transition-colors
                    ${form.task_type === k ? 'border-blue-400 bg-blue-50' : 'border-slate-200 hover:bg-slate-50'}`}>
                  <input type="radio" name="task_type" className="mt-0.5 accent-blue-600"
                    checked={form.task_type === k}
                    onChange={() => setForm(f => ({ ...f, task_type: k, prompt_template: '' }))} />
                  <div className="min-w-0 flex-1">
                    <p className="text-sm font-medium text-slate-700 flex items-center gap-2">
                      {label}
                      {srcType && (
                        <span className="text-[11px] font-normal text-slate-400">
                          源：{srcType === 'guideline' ? '指南' : '病例报告'}
                        </span>
                      )}
                      {!srcType && (
                        <span className="text-[11px] font-normal text-purple-500">源：上游 job</span>
                      )}
                    </p>
                    <p className="text-xs text-slate-400 mt-0.5 leading-relaxed">{desc}</p>
                  </div>
                </label>
              ))}
            </div>
            {!isAugment && docStats != null && (
              <p className={`mt-2 text-xs flex items-center gap-1 ${availableDocs > 0 ? 'text-slate-500' : 'text-red-500'}`}>
                <span>当前可用源文档：<b className={availableDocs > 0 ? 'text-slate-700' : 'text-red-600'}>{availableDocs}</b> 条</span>
                {availableDocs === 0 && <span>· 请先到「文档管理」加载</span>}
                {availableDocs > 0 && form.doc_limit && parseInt(form.doc_limit) < availableDocs && (
                  <span className="text-amber-600">· 本次将仅处理前 {form.doc_limit} 条</span>
                )}
              </p>
            )}
          </div>

          {/* guideline_synth: N 配置 */}
          {isGuidelineSynth && (
            <div className="p-3 bg-teal-50/50 border border-teal-100 rounded-lg">
              <label className="text-xs text-slate-600 mb-1 block font-medium">
                每篇指南合成虚拟患者数（N）
              </label>
              <input
                type="number"
                min="2" max="30" step="1"
                className="w-32 px-3 py-1.5 text-sm border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-teal-400"
                value={form.n_per_doc}
                onChange={e => setForm(f => ({ ...f, n_per_doc: e.target.value }))}
              />
              <p className="text-[11px] text-slate-400 mt-1">
                建议 6~12。N 越大产出越多但单次 LLM 调用越慢且 max_tokens 占用越高（自动按 N 放大）
              </p>
            </div>
          )}

          {/* augment: 源 job 选 + 策略勾 */}
          {isAugment && (
            <div className="p-3 bg-purple-50/50 border border-purple-100 rounded-lg space-y-3">
              <div>
                <label className="text-xs text-slate-600 mb-1 block font-medium">源任务 *</label>
                <select
                  className="w-full px-3 py-1.5 text-sm border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-400"
                  value={form.source_job_id}
                  onChange={e => setForm(f => ({ ...f, source_job_id: e.target.value }))}>
                  <option value="">请选择一个已完成的抽取任务...</option>
                  {augmentableJobs.map(j => (
                    <option key={j.id} value={j.id}>
                      #{j.id} {j.name} （{j.task_type}, {j.processed_docs} 文档）
                    </option>
                  ))}
                </select>
                {augmentableJobs.length === 0 && (
                  <p className="text-[11px] text-amber-600 mt-1">
                    暂无已完成的抽取任务可作源；请先跑一个 case_extract / guideline_synth / case_reasoning
                  </p>
                )}
              </div>

              <div>
                <label className="text-xs text-slate-600 mb-1 block font-medium">增强策略 *</label>
                <div className="space-y-1.5">
                  {augStrategies.map(s => (
                    <label key={s.name}
                      className={`flex items-start gap-2 p-2 rounded border cursor-pointer transition-colors text-xs
                        ${form.augment_strategies.includes(s.name) ? 'border-purple-400 bg-purple-50' : 'border-slate-200 hover:bg-slate-50'}`}>
                      <input type="checkbox" className="mt-0.5 accent-purple-600"
                        checked={form.augment_strategies.includes(s.name)}
                        onChange={() => toggleStrategy(s.name)} />
                      <div className="min-w-0 flex-1">
                        <p className="text-slate-700 font-medium">
                          {s.label}
                          {s.recommended && <span className="ml-1.5 text-[10px] text-green-600">推荐</span>}
                        </p>
                        <p className="text-[11px] text-slate-400 mt-0.5">{s.desc}</p>
                      </div>
                    </label>
                  ))}
                </div>
              </div>

              <div className="grid grid-cols-2 gap-2">
                <div>
                  <label className="text-xs text-slate-600 mb-1 block">源实例上限</label>
                  <input type="number" min="1" step="50"
                    className="w-full px-3 py-1.5 text-sm border border-slate-200 rounded-lg focus:outline-none"
                    value={form.max_source_instances}
                    onChange={e => setForm(f => ({ ...f, max_source_instances: e.target.value }))} />
                </div>
                <div>
                  <label className="text-xs text-slate-600 mb-1 block">每策略变体数</label>
                  <input type="number" min="1" max="5" step="1"
                    className="w-full px-3 py-1.5 text-sm border border-slate-200 rounded-lg focus:outline-none"
                    value={form.variants_per_strategy}
                    onChange={e => setForm(f => ({ ...f, variants_per_strategy: e.target.value }))} />
                </div>
              </div>
              <p className="text-[11px] text-slate-400">
                预计产出：约 <b>{Math.min(Number(form.max_source_instances) || 0, augmentableJobs.find(j => String(j.id) === String(form.source_job_id))?.processed_docs || 0) * form.augment_strategies.length * (Number(form.variants_per_strategy) || 1)}</b> 条变体（每条变体 = 1 次 LLM 调用）
              </p>
            </div>
          )}

          {!isAugment && (
            <div>
              <label className="text-xs text-slate-500 mb-1 block">
                处理文档数量上限
                <span className="ml-1 text-slate-400">（留空则处理全部文档）</span>
              </label>
              <input
                type="number"
                min="1"
                className="w-full px-3 py-2 text-sm border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                value={form.doc_limit}
                onChange={e => setForm(f => ({ ...f, doc_limit: e.target.value }))}
                placeholder={form.task_type === 'case_reasoning' ? '推理合成耗时较长，建议先设 10~30 条试跑' : '例：100（测试时建议先设置小数量）'}
              />
            </div>
          )}

          <div className="border-t border-slate-100 pt-3">
            <p className="text-xs font-medium text-slate-600 mb-2">LLM 配置</p>
            <div className="flex gap-1 mb-3">
              {[['assistant', '使用已配置助手'], ['manual', '手动填写参数']].map(([k, label]) => (
                <button key={k} type="button" onClick={() => setForm(f => ({ ...f, llm_mode: k }))}
                  className={`px-3 py-1.5 text-xs rounded-lg font-medium transition-colors
                    ${form.llm_mode === k ? 'bg-blue-100 text-blue-700' : 'text-slate-500 hover:bg-slate-100'}`}>
                  {label}
                </button>
              ))}
            </div>

            {useAssistant ? (
              <div>
                <select className="w-full px-3 py-2 text-sm border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                  value={form.assistant_id}
                  onChange={e => setForm(f => ({ ...f, assistant_id: e.target.value }))}>
                  <option value="">请选择助手</option>
                  {assistantList.map(a => (
                    <option key={a.id} value={a.id}>
                      {a.type === 'local' ? '🖥' : '☁'} {a.name} ({a.model_name})
                    </option>
                  ))}
                </select>
                {assistantList.length === 0 && (
                  <p className="mt-2 text-xs text-amber-600">
                    暂无运行中的助手，请到「临床助手 → 助手管理」配置并启动
                  </p>
                )}
              </div>
            ) : (
              <div className="space-y-3">
                <div>
                  <label className="text-xs text-slate-500 mb-1 block">Base URL</label>
                  <input className="w-full px-3 py-2 text-sm border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                    value={form.base_url}
                    onChange={e => setForm(f => ({ ...f, base_url: e.target.value }))}
                    placeholder="例：https://api.openai.com/v1 或 http://localhost:8000/v1" />
                  {form.base_url && (
                    <p className={`mt-1 text-xs ${isLocal ? 'text-green-600' : 'text-slate-500'}`}>
                      {isLocal ? '✓ 本地端点：api_key 可为空' : '远程端点：api_key 必填'}
                    </p>
                  )}
                </div>
                <div>
                  <label className="text-xs text-slate-500 mb-1 block">Model name</label>
                  <input className="w-full px-3 py-2 text-sm border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                    value={form.model}
                    onChange={e => setForm(f => ({ ...f, model: e.target.value }))}
                    placeholder="例：gpt-4o-mini / qwen2.5:7b" />
                </div>
                <div>
                  <label className="text-xs text-slate-500 mb-1 block">
                    API Key {isLocal && <span className="text-slate-400">（本地端点可空）</span>}
                  </label>
                  <div className="relative">
                    <input
                      type={showKey ? 'text' : 'password'}
                      className={`w-full px-3 py-2 pr-16 text-sm border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 ${needKey ? 'border-red-300' : 'border-slate-200'}`}
                      value={form.api_key}
                      onChange={e => setForm(f => ({ ...f, api_key: e.target.value }))}
                      placeholder="sk-..."
                      autoComplete="off"
                    />
                    <button type="button" onClick={() => setShowKey(v => !v)}
                      className="absolute right-2 top-1/2 -translate-y-1/2 text-xs text-slate-400 hover:text-slate-600">
                      {showKey ? '隐藏' : '显示'}
                    </button>
                  </div>
                </div>
              </div>
            )}
          </div>

          {!isAugment && (
            <div className="border-t border-slate-100 pt-3">
              <div className="flex items-center justify-between mb-1">
                <label className="text-xs text-slate-500">
                  提示词模板（留空使用默认 · 当前默认：{({case_extract:'病例抽取', guideline_synth:'指南合成', case_reasoning:'临床推理合成'})[form.task_type]}）
                </label>
                <button className="text-xs text-blue-600 hover:underline"
                  onClick={() => setForm(f => ({ ...f, prompt_template: defaults[defaultTemplateKey] || '' }))}>
                  填入默认模板
                </button>
              </div>
              <textarea className="w-full px-3 py-2 text-xs border border-slate-200 rounded-lg focus:outline-none font-mono h-28 resize-none"
                value={form.prompt_template} onChange={e => setForm(f => ({ ...f, prompt_template: e.target.value }))}
                placeholder={isGuidelineSynth ? '使用 {content} 与 {n_patients} 作为占位符' : '使用 {content} 作为文档内容占位符'} />
            </div>
          )}
        </div>

        {err && <p className="mt-3 text-xs text-red-500">{err}</p>}

        <div className="flex justify-end gap-2 mt-5">
          <button onClick={onClose} className="px-4 py-2 text-sm border border-slate-200 rounded-lg hover:bg-slate-50">取消</button>
          <button onClick={handleSubmit} disabled={loading || !form.name || needKey}
            className="px-4 py-2 text-sm bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50">
            {loading ? '创建中...' : '创建并运行'}
          </button>
        </div>
      </div>
    </div>
  )
}

const STRATEGY_LABEL = {
  case_direct:      { label: '病例直出',       cls: 'bg-sky-100 text-sky-700' },
  guideline_synth:  { label: '指南合成',       cls: 'bg-teal-100 text-teal-700' },
  aug_paraphrase:   { label: '改写增强',       cls: 'bg-purple-100 text-purple-700' },
  aug_distractor:   { label: '干扰注入',       cls: 'bg-orange-100 text-orange-700' },
  aug_hardneg:      { label: '困难负样本',     cls: 'bg-rose-100 text-rose-700' },
  aug_cot:          { label: 'CoT 扩写',       cls: 'bg-indigo-100 text-indigo-700' },
  aug_comorbidity:  { label: '合并症叠加',     cls: 'bg-amber-100 text-amber-700' },
}

function InstancesPanel({ jobId }) {
  const [data, setData] = useState({ items: [], total: 0 })
  const [page, setPage] = useState(1)
  const [expanded, setExpanded] = useState({})

  useEffect(() => {
    extraction.listInstances({ job_id: jobId, page, page_size: 10 }).then(r => setData(r.data))
  }, [jobId, page])

  const toggleApprove = async (item) => {
    await extraction.approveInstance(item.id)
    setData(d => ({
      ...d,
      items: d.items.map(i => i.id === item.id ? { ...i, is_approved: !i.is_approved } : i)
    }))
  }

  const toggleExpand = (id) => setExpanded(e => ({ ...e, [id]: !e[id] }))

  const truncate = (s, n, open) =>
    typeof s === 'string' && s.length > n && !open ? s.slice(0, n) + '…' : s

  return (
    <div className="mt-4 border-t border-slate-100 pt-4">
      <p className="text-xs text-slate-500 mb-2">诊断实例（共 {data.total} 条）</p>
      <div className="space-y-2 max-h-96 overflow-y-auto">
        {data.items.map(item => {
          const meta = STRATEGY_LABEL[item.synthesis_strategy]
            || { label: item.synthesis_strategy || '?', cls: 'bg-slate-100 text-slate-600' }
          const isOpen = expanded[item.id]
          return (
            <div key={item.id} className="p-3 bg-slate-50 rounded-lg text-xs">
              <div className="flex items-center justify-between mb-1.5 flex-wrap gap-1">
                <div className="flex items-center gap-1.5 flex-wrap">
                  <span className={`px-1.5 py-0.5 rounded text-xs font-medium ${meta.cls}`}>{meta.label}</span>
                  {item.diagnosis_label && (
                    <span className="px-1.5 py-0.5 bg-slate-200 text-slate-700 rounded text-[10px]">
                      {item.diagnosis_label}
                    </span>
                  )}
                  {item.specialty && (
                    <span className="text-[10px] text-slate-400">{item.specialty}</span>
                  )}
                  {item.parent_instance_id && (
                    <span className="text-[10px] text-slate-400">← #{item.parent_instance_id}</span>
                  )}
                </div>
                <button onClick={() => toggleApprove(item)}
                  className={`text-xs ${item.is_approved ? 'text-green-600 font-medium' : 'text-slate-400'} hover:text-green-600`}>
                  {item.is_approved ? '✓ 已审核' : '审核'}
                </button>
              </div>
              <div className="space-y-1.5">
                <div>
                  <span className="text-slate-400 text-[11px] mr-1">presentation:</span>
                  <span className="text-slate-700 whitespace-pre-wrap">{truncate(item.presentation, 280, isOpen)}</span>
                </div>
                <div>
                  <span className="text-slate-400 text-[11px] mr-1">answer:</span>
                  <span className="text-slate-700 whitespace-pre-wrap">{truncate(item.answer, 400, isOpen)}</span>
                </div>
                {((item.presentation?.length || 0) > 280 || (item.answer?.length || 0) > 400) && (
                  <button onClick={() => toggleExpand(item.id)}
                    className="text-[11px] text-blue-600 hover:underline">
                    {isOpen ? '收起' : '展开全部'}
                  </button>
                )}
              </div>
            </div>
          )
        })}
      </div>
      {data.total > 10 && (
        <div className="flex items-center justify-between mt-3 text-xs text-slate-500">
          <button disabled={page <= 1} onClick={() => setPage(p => p - 1)}
            className="px-2 py-1 border border-slate-200 rounded hover:bg-slate-50 disabled:opacity-40">上一页</button>
          <span>第 {page} 页 / 共 {Math.ceil(data.total / 10)} 页</span>
          <button disabled={page >= Math.ceil(data.total / 10)} onClick={() => setPage(p => p + 1)}
            className="px-2 py-1 border border-slate-200 rounded hover:bg-slate-50 disabled:opacity-40">下一页</button>
        </div>
      )}
    </div>
  )
}

function JobCard({ job, onRefresh }) {
  const [expanded, setExpanded] = useState(false)
  const [actionLoading, setActionLoading] = useState(false)
  const [confirmDelete, setConfirmDelete] = useState(false)

  const cfg = STATUS_CONFIG[job.status] || STATUS_CONFIG.pending
  const Icon = cfg.icon
  const progress = job.total_docs > 0 ? Math.round((job.processed_docs / job.total_docs) * 100) : 0

  const handlePause = async () => {
    setActionLoading(true)
    try {
      await extraction.cancelJob(job.id)
      onRefresh()
    } catch (e) { console.error(e) }
    setActionLoading(false)
  }

  const handleRestart = async () => {
    setActionLoading(true)
    try {
      await extraction.restartJob(job.id)
      onRefresh()
    } catch (e) { console.error(e) }
    setActionLoading(false)
  }

  const handleDelete = async () => {
    if (!confirmDelete) { setConfirmDelete(true); return }
    setActionLoading(true)
    try {
      await extraction.deleteJob(job.id)
      onRefresh()
    } catch (e) { console.error(e) }
    setActionLoading(false)
    setConfirmDelete(false)
  }

  return (
    <div className="bg-white rounded-xl shadow-sm border border-slate-100 p-5">
      <div className="flex items-start justify-between gap-3">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1 flex-wrap">
            <span className="font-medium text-slate-800 truncate">{job.name}</span>
            <span className={`flex items-center gap-1 px-2 py-0.5 rounded text-xs shrink-0 ${cfg.color}`}>
              <Icon size={10} /> {cfg.label}
            </span>
            {job.task_type === 'case_reasoning' && (
              <span className="px-2 py-0.5 bg-rose-100 text-rose-700 rounded text-xs shrink-0">
                临床推理合成
              </span>
            )}
            {job.task_type === 'guideline_synth' && (
              <span className="px-2 py-0.5 bg-teal-100 text-teal-700 rounded text-xs shrink-0">
                指南合成
              </span>
            )}
            {job.task_type === 'augment' && (
              <span className="px-2 py-0.5 bg-purple-100 text-purple-700 rounded text-xs shrink-0">
                增强 ← #{job.source_job_id}
              </span>
            )}
          </div>
          <div className="text-xs text-slate-400 flex flex-wrap gap-x-3 gap-y-0.5">
            <span>类型：{job.document_type === 'case_report' ? '病例报告' : job.document_type === 'guideline' ? '指南/共识' : '全部'}</span>
            <span>模型：{job.model || '默认'}</span>
            {job.base_url && <span title={job.base_url} className="truncate max-w-[18rem]">端点：{job.base_url}</span>}
            {job.has_api_key === false && job.base_url && <span className="text-amber-600">无 api_key</span>}
            {job.doc_limit && <span className="text-orange-500">限制：{job.doc_limit} 条</span>}
            <span>创建：{job.created_at?.slice(0, 10)}</span>
          </div>
          {job.total_docs > 0 && (
            <div className="mt-3">
              <div className="flex justify-between text-xs text-slate-500 mb-1">
                <span>
                  {job.processed_docs} / {job.total_docs} 文档
                  {job.failed_docs > 0 && <span className="text-red-400 ml-2">失败 {job.failed_docs}</span>}
                </span>
                <span>{progress}%</span>
              </div>
              <div className="h-1.5 bg-slate-100 rounded-full overflow-hidden">
                <div className={`h-full rounded-full transition-all ${job.status === 'paused' ? 'bg-yellow-400' : 'bg-blue-500'}`}
                  style={{ width: `${progress}%` }} />
              </div>
            </div>
          )}
          {job.error_message && <p className="mt-2 text-xs text-red-500 break-all">{job.error_message}</p>}
        </div>

        {/* 操作按钮区 */}
        <div className="flex items-center gap-1.5 shrink-0">
          {job.status === 'running' && (
            <button
              onClick={handlePause}
              disabled={actionLoading}
              title="暂停任务"
              className="flex items-center gap-1 px-2.5 py-1.5 text-xs border border-yellow-300 text-yellow-700 bg-yellow-50 rounded-lg hover:bg-yellow-100 disabled:opacity-50"
            >
              <Pause size={12} /> 暂停
            </button>
          )}
          {job.status !== 'running' && (
            <button
              onClick={handleRestart}
              disabled={actionLoading}
              title="重新运行任务"
              className="flex items-center gap-1 px-2.5 py-1.5 text-xs border border-green-300 text-green-700 bg-green-50 rounded-lg hover:bg-green-100 disabled:opacity-50"
            >
              <RotateCcw size={12} /> 重启
            </button>
          )}
          <button
            onClick={handleDelete}
            disabled={actionLoading}
            title={confirmDelete ? '再次点击确认删除' : '删除任务'}
            className={`flex items-center gap-1 px-2.5 py-1.5 text-xs rounded-lg disabled:opacity-50 transition-colors ${
              confirmDelete
                ? 'border border-red-500 text-white bg-red-500 hover:bg-red-600'
                : 'border border-slate-200 text-slate-500 hover:bg-red-50 hover:text-red-500 hover:border-red-200'
            }`}
            onBlur={() => setConfirmDelete(false)}
          >
            <Trash2 size={12} /> {confirmDelete ? '确认删除' : '删除'}
          </button>
          <button
            onClick={() => setExpanded(v => !v)}
            className="flex items-center gap-1 px-2.5 py-1.5 text-xs border border-slate-200 text-slate-600 rounded-lg hover:bg-slate-50"
          >
            {expanded ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
            {expanded ? '收起' : '结果'}
          </button>
        </div>
      </div>

      {expanded && <InstancesPanel jobId={job.id} />}
    </div>
  )
}

export default function Extraction() {
  const [jobs, setJobs] = useState([])
  const [showCreate, setShowCreate] = useState(false)
  const [loading, setLoading] = useState(false)
  const pollRef = useRef(null)

  const load = async () => {
    setLoading(true)
    try {
      const r = await extraction.listJobs()
      setJobs(r.data)
    } catch {}
    setLoading(false)
  }

  // 有运行中的任务时自动轮询刷新
  useEffect(() => {
    load()
  }, [])

  useEffect(() => {
    const hasRunning = jobs.some(j => j.status === 'running')
    if (hasRunning && !pollRef.current) {
      pollRef.current = setInterval(load, 5000)
    } else if (!hasRunning && pollRef.current) {
      clearInterval(pollRef.current)
      pollRef.current = null
    }
    return () => {
      if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null }
    }
  }, [jobs])

  const runningCount = jobs.filter(j => j.status === 'running').length

  return (
    <div className="p-8">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-xl font-bold text-slate-800">知识抽取</h1>
          <p className="text-sm text-slate-500 mt-0.5">使用大模型从文档中抽取结构化医学知识</p>
        </div>
        <div className="flex gap-2">
          <button onClick={load} className="flex items-center gap-1.5 px-3 py-2 text-sm border border-slate-200 rounded-lg hover:bg-slate-50">
            <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
            刷新{runningCount > 0 && <span className="ml-1 text-xs text-blue-600">（{runningCount} 个运行中，自动刷新）</span>}
          </button>
          <button onClick={() => setShowCreate(true)} className="flex items-center gap-1.5 px-3 py-2 text-sm bg-blue-600 text-white rounded-lg hover:bg-blue-700">
            <Plus size={14} /> 新建任务
          </button>
        </div>
      </div>

      <div className="space-y-3">
        {jobs.map(job => (
          <JobCard key={job.id} job={job} onRefresh={load} />
        ))}
        {!loading && jobs.length === 0 && (
          <div className="text-center py-16 text-slate-400">
            <div className="text-4xl mb-3">🔬</div>
            <p>暂无抽取任务，点击「新建任务」开始</p>
            <p className="text-xs mt-1 text-slate-300">建议先设置较小的文档数量（如 10~50 条）进行测试</p>
          </div>
        )}
      </div>

      {showCreate && <CreateJobModal onClose={() => setShowCreate(false)} onCreated={load} existingJobs={jobs} />}
    </div>
  )
}
