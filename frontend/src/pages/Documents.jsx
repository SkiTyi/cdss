import { useEffect, useState } from 'react'
import { documents } from '../api/client'
import { Search, Download, RefreshCw, X, FileText, Loader2, Trash2, AlertTriangle, Filter, Database } from 'lucide-react'
import MarkdownView from '../components/MarkdownView'

const TYPE_LABELS = { case_report: '病例报告', guideline: '临床指南' }
const STATUS_COLORS = { pending: 'bg-slate-100 text-slate-600', extracted: 'bg-green-100 text-green-700' }

function DocumentViewer({ docId, onClose }) {
  const [doc, setDoc] = useState(null)
  const [loading, setLoading] = useState(true)
  const [err, setErr] = useState('')

  useEffect(() => {
    if (!docId) return
    setLoading(true); setErr(''); setDoc(null)
    documents.get(docId)
      .then(r => setDoc(r.data))
      .catch(e => setErr(e?.response?.data?.detail || '加载失败'))
      .finally(() => setLoading(false))
  }, [docId])

  useEffect(() => {
    const onKey = (e) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  return (
    <div className="fixed inset-0 bg-black/40 z-50 flex items-stretch justify-end" onClick={onClose}>
      <div
        className="bg-white w-full max-w-3xl h-full shadow-2xl flex flex-col animate-in slide-in-from-right"
        onClick={e => e.stopPropagation()}
      >
        <div className="px-5 py-4 border-b border-slate-100 flex items-start gap-3">
          <FileText size={18} className="text-blue-600 mt-0.5 shrink-0" />
          <div className="flex-1 min-w-0">
            <p className="font-semibold text-slate-800 truncate">{doc?.title || '加载中...'}</p>
            {doc && (
              <div className="mt-0.5 text-xs text-slate-400 flex flex-wrap gap-x-3 gap-y-0.5">
                <span>类型：{TYPE_LABELS[doc.type] || doc.type}</span>
                <span>状态：{doc.status === 'extracted' ? '已抽取' : '待处理'}</span>
                {doc.created_at && <span>创建：{doc.created_at.slice(0, 10)}</span>}
                {doc.source_path && <span className="truncate">源：{doc.source_path}</span>}
              </div>
            )}
          </div>
          <button onClick={onClose}
            className="p-1.5 text-slate-400 hover:text-slate-700 hover:bg-slate-100 rounded-lg shrink-0">
            <X size={16} />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto px-6 py-5">
          {loading && (
            <div className="flex items-center justify-center py-16 text-slate-400">
              <Loader2 size={20} className="animate-spin mr-2" /> 正在加载文档内容...
            </div>
          )}
          {err && <p className="text-red-500 text-sm">{err}</p>}
          {doc && !loading && (
            doc.content
              ? <MarkdownView source={doc.content} />
              : <p className="text-slate-400 text-sm py-8 text-center">该文档没有可显示的内容。</p>
          )}
        </div>

        {doc && doc.content && (
          <div className="px-5 py-2 border-t border-slate-100 text-xs text-slate-400 text-right">
            字符数 {doc.content.length.toLocaleString()}
          </div>
        )}
      </div>
    </div>
  )
}


function ConfirmDialog({ title, message, detail, confirmLabel = '确认', tone = 'danger', loading, onConfirm, onCancel }) {
  const toneCls = tone === 'danger'
    ? 'bg-red-600 hover:bg-red-700'
    : tone === 'warning'
    ? 'bg-amber-600 hover:bg-amber-700'
    : 'bg-blue-600 hover:bg-blue-700'
  const iconBg = tone === 'danger'
    ? 'bg-red-50 text-red-600'
    : tone === 'warning'
    ? 'bg-amber-50 text-amber-600'
    : 'bg-blue-50 text-blue-600'

  useEffect(() => {
    const onKey = (e) => { if (e.key === 'Escape' && !loading) onCancel() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onCancel, loading])

  return (
    <div className="fixed inset-0 bg-black/40 z-50 flex items-center justify-center p-4"
      onClick={() => !loading && onCancel()}>
      <div className="bg-white rounded-xl shadow-xl w-full max-w-md p-5" onClick={e => e.stopPropagation()}>
        <div className="flex items-start gap-3">
          <div className={`p-2 rounded-lg shrink-0 ${iconBg}`}>
            <AlertTriangle size={18} />
          </div>
          <div className="flex-1 min-w-0">
            <h3 className="text-base font-semibold text-slate-800">{title}</h3>
            <p className="text-sm text-slate-600 mt-1 leading-relaxed">{message}</p>
            {detail && (
              <p className="text-xs text-slate-400 mt-2 leading-relaxed">{detail}</p>
            )}
          </div>
        </div>
        <div className="flex justify-end gap-2 mt-5">
          <button
            onClick={onCancel}
            disabled={loading}
            className="px-4 py-2 text-sm border border-slate-200 rounded-lg hover:bg-slate-50 disabled:opacity-50"
          >取消</button>
          <button
            onClick={onConfirm}
            disabled={loading}
            className={`px-4 py-2 text-sm text-white rounded-lg disabled:opacity-50 ${toneCls}`}
          >
            {loading ? '处理中...' : confirmLabel}
          </button>
        </div>
      </div>
    </div>
  )
}


export default function Documents() {
  const [data, setData] = useState({ items: [], total: 0 })
  const [params, setParams] = useState({ page: 1, page_size: 20, type: '', search: '', min_content_length: 0 })
  const [stats, setStats] = useState(null)
  const [loading, setLoading] = useState(false)
  const [loadMsg, setLoadMsg] = useState('')
  const [viewingId, setViewingId] = useState(null)
  // selected ids on the CURRENT page (cleared on page/filter change)
  const [selected, setSelected] = useState(() => new Set())
  // confirm describes the pending action; null when no dialog open
  const [confirm, setConfirm] = useState(null)
  const [confirmLoading, setConfirmLoading] = useState(false)

  const fetchList = async (p = params) => {
    setLoading(true)
    try {
      const res = await documents.list({
        ...p,
        type: p.type || undefined,
        search: p.search || undefined,
        min_content_length: p.min_content_length > 0 ? p.min_content_length : undefined,
      })
      setData(res.data)
      setSelected(new Set())  // clear selection whenever page reloads
    } catch {}
    setLoading(false)
  }

  const fetchStats = () => documents.stats().then(r => setStats(r.data)).catch(() => {})

  useEffect(() => {
    fetchList()
    fetchStats()
  }, [])

  const setParam = (key, val) => {
    const next = { ...params, [key]: val, page: 1 }
    setParams(next)
    fetchList(next)
  }

  const toggleSelect = (id) => {
    setSelected(s => {
      const n = new Set(s)
      n.has(id) ? n.delete(id) : n.add(id)
      return n
    })
  }
  const toggleSelectAll = () => {
    const allOnPage = data.items.map(d => d.id)
    const allSelected = allOnPage.length > 0 && allOnPage.every(id => selected.has(id))
    setSelected(s => {
      const n = new Set(s)
      if (allSelected) {
        allOnPage.forEach(id => n.delete(id))
      } else {
        allOnPage.forEach(id => n.add(id))
      }
      return n
    })
  }

  // ── action runners (called from confirm dialog) ─────────────────────────
  const runConfirm = async () => {
    if (!confirm) return
    setConfirmLoading(true)
    try {
      const res = await confirm.run()
      const deleted = res?.data?.deleted
      const msg = confirm.successMsg
        ? confirm.successMsg(deleted)
        : (deleted != null ? `已删除 ${deleted} 条` : '操作完成')
      setLoadMsg(msg)
      setConfirm(null)
      await fetchList()
      await fetchStats()
    } catch (e) {
      setLoadMsg('操作失败：' + (e?.response?.data?.detail || e.message || '未知错误'))
      setConfirm(null)
    }
    setConfirmLoading(false)
  }

  // ── confirm-dialog launchers ────────────────────────────────────────────
  const askDeleteOne = (doc) => setConfirm({
    title: '删除该文档？',
    message: `「${doc.title}」将从数据库中移除。`,
    detail: '仅从数据库中移除，磁盘上的原始文件不会被删除；之后可点击「加载数据文件」重新导入。',
    confirmLabel: '删除',
    tone: 'danger',
    run: () => documents.delete(doc.id),
    successMsg: () => `已删除「${doc.title}」`,
  })

  const askDeleteBulk = () => {
    const ids = Array.from(selected)
    setConfirm({
      title: `删除选中的 ${ids.length} 条文档？`,
      message: '所选文档将从数据库中移除。',
      detail: '仅从数据库中移除，磁盘上的原始文件不会被删除。',
      confirmLabel: `删除 ${ids.length} 条`,
      tone: 'danger',
      run: () => documents.deleteBulk(ids),
    })
  }

  const askDeleteShort = (threshold) => setConfirm({
    title: `永久过滤字符数 < ${threshold} 的文档？`,
    message: `所有正文不足 ${threshold} 字符的文档将从数据库中移除。`,
    detail: '不同于上方的「最小字符数」筛选条件（仅过滤当前视图），此操作会真正从文档数据库中删除这些短文档，确保后续知识抽取不会再读取它们。磁盘上的原始文件不会被删除。',
    confirmLabel: '执行删除',
    tone: 'warning',
    run: () => documents.deleteShort(threshold, params.type || null),
  })

  const askReset = () => setConfirm({
    title: '清空整个文档数据库？',
    message: '当前所有文档都将从数据库中移除。',
    detail: '此操作不可撤销。磁盘上的原始文件不会被删除；之后可点击「加载数据文件」从零开始重新导入。',
    confirmLabel: '清空数据库',
    tone: 'danger',
    run: () => documents.reset(),
  })

  const askReload = () => setConfirm({
    title: '重新加载磁盘上的数据文件？',
    message: '这会从磁盘扫描原始数据文件并补全文档数据库。',
    detail: '注意：磁盘上仍然存在的原始文件如果之前被你删除过，将再次被导入回数据库（恢复成未过滤的状态）。已经存在的文档会被跳过、不会重复。',
    confirmLabel: '开始加载',
    tone: 'warning',
    run: async () => {
      const r = await documents.load()
      // load() returns immediately; the actual import runs as a background
      // task. Schedule a delayed refresh so the page picks up the new rows.
      setTimeout(() => { fetchList(); fetchStats() }, 3000)
      return r
    },
    successMsg: () => '数据加载已启动，请稍后刷新查看结果',
  })

  // ── derived helpers ─────────────────────────────────────────────────────
  const allOnPageSelected = data.items.length > 0
    && data.items.every(d => selected.has(d.id))
  const someOnPageSelected = data.items.some(d => selected.has(d.id))

  return (
    <div className="p-8">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-xl font-bold text-slate-800">文档管理</h1>
          <p className="text-sm text-slate-500 mt-0.5">共 {data.total} 条文档</p>
        </div>
        <div className="flex gap-2 flex-wrap">
          <button onClick={() => fetchList()}
            className="flex items-center gap-1.5 px-3 py-2 text-sm border border-slate-200 rounded-lg hover:bg-slate-50">
            <RefreshCw size={14} className={loading ? 'animate-spin' : ''} /> 刷新
          </button>
          <button onClick={askReset}
            className="flex items-center gap-1.5 px-3 py-2 text-sm border border-red-200 text-red-600 rounded-lg hover:bg-red-50">
            <Database size={14} /> 清空数据库
          </button>
          <button onClick={askReload}
            className="flex items-center gap-1.5 px-3 py-2 text-sm bg-blue-600 text-white rounded-lg hover:bg-blue-700">
            <Download size={14} /> 加载数据文件
          </button>
        </div>
      </div>

      {loadMsg && (
        <div className="mb-4 p-3 bg-blue-50 text-blue-700 text-sm rounded-lg flex items-center justify-between">
          <span>{loadMsg}</span>
          <button onClick={() => setLoadMsg('')} className="text-blue-500 hover:text-blue-700">
            <X size={14} />
          </button>
        </div>
      )}

      {stats && (stats.short_lt_100 > 0 || stats.short_lt_500 > 0) && (
        <div className="mb-4 p-3 bg-amber-50 text-amber-700 text-xs rounded-lg flex items-center gap-3 flex-wrap">
          <span>⚠ 当前数据库中内容偏短文档：</span>
          <span>&lt; 100 字符：<b>{stats.short_lt_100}</b> 条</span>
          <span>·</span>
          <span>&lt; 500 字符：<b>{stats.short_lt_500}</b> 条</span>
          <div className="ml-auto flex gap-1.5">
            {stats.short_lt_100 > 0 && (
              <button onClick={() => askDeleteShort(100)}
                className="flex items-center gap-1 px-2 py-1 bg-amber-100 hover:bg-amber-200 text-amber-700 rounded text-xs">
                <Trash2 size={11} /> 一键删除 &lt; 100
              </button>
            )}
            {stats.short_lt_500 > 0 && (
              <button onClick={() => askDeleteShort(500)}
                className="flex items-center gap-1 px-2 py-1 bg-amber-100 hover:bg-amber-200 text-amber-700 rounded text-xs">
                <Trash2 size={11} /> 一键删除 &lt; 500
              </button>
            )}
          </div>
        </div>
      )}

      <div className="bg-white rounded-xl shadow-sm border border-slate-100">
        <div className="p-4 border-b border-slate-100 flex gap-3 flex-wrap items-center">
          <div className="relative flex-1 min-w-[200px]">
            <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
            <input
              className="w-full pl-8 pr-3 py-2 text-sm border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              placeholder="搜索标题..."
              value={params.search}
              onChange={e => setParam('search', e.target.value)}
            />
          </div>
          <select
            className="px-3 py-2 text-sm border border-slate-200 rounded-lg focus:outline-none"
            value={params.type}
            onChange={e => setParam('type', e.target.value)}
          >
            <option value="">全部类型</option>
            <option value="case_report">病例报告</option>
            <option value="guideline">临床指南</option>
          </select>
          <div className="flex items-center gap-2 px-3 py-2 border border-slate-200 rounded-lg text-sm"
            title="仅作为视图筛选条件，不会修改数据库">
            <Filter size={12} className="text-slate-400" />
            <span className="text-slate-500 text-xs whitespace-nowrap">视图筛选 ≥</span>
            <input
              type="number"
              min="0"
              step="50"
              className="w-20 outline-none border-none bg-transparent text-slate-700"
              value={params.min_content_length}
              onChange={e => setParam('min_content_length', Math.max(0, parseInt(e.target.value || '0')))}
              placeholder="0"
            />
            <span className="text-xs text-slate-400">字符</span>
            {params.min_content_length > 0 && (
              <button
                onClick={() => setParam('min_content_length', 0)}
                className="text-xs text-slate-400 hover:text-red-500"
                title="清除筛选"
              >×</button>
            )}
          </div>
        </div>

        {/* 选中操作栏 */}
        {selected.size > 0 && (
          <div className="px-4 py-2 bg-blue-50 border-b border-blue-100 flex items-center justify-between text-sm">
            <span className="text-blue-700">已选中 <b>{selected.size}</b> 条文档</span>
            <div className="flex items-center gap-2">
              <button onClick={() => setSelected(new Set())}
                className="text-xs text-slate-500 hover:text-slate-700">取消选择</button>
              <button onClick={askDeleteBulk}
                className="flex items-center gap-1 px-3 py-1.5 text-xs bg-red-600 text-white rounded-lg hover:bg-red-700">
                <Trash2 size={11} /> 批量删除
              </button>
            </div>
          </div>
        )}

        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-slate-100 text-left text-slate-500">
              <th className="px-4 py-3 w-8">
                <input
                  type="checkbox"
                  className="accent-blue-600 cursor-pointer"
                  checked={allOnPageSelected}
                  ref={el => { if (el) el.indeterminate = !allOnPageSelected && someOnPageSelected }}
                  onChange={toggleSelectAll}
                  title="全选本页"
                />
              </th>
              <th className="px-4 py-3 font-medium">标题</th>
              <th className="px-4 py-3 font-medium w-24">类型</th>
              <th className="px-4 py-3 font-medium w-24">字符数</th>
              <th className="px-4 py-3 font-medium w-24">状态</th>
              <th className="px-4 py-3 font-medium w-36">创建时间</th>
              <th className="px-4 py-3 font-medium w-28 text-right">操作</th>
            </tr>
          </thead>
          <tbody>
            {data.items.map(doc => (
              <tr key={doc.id}
                onClick={() => setViewingId(doc.id)}
                className={`border-b border-slate-50 hover:bg-blue-50/40 cursor-pointer transition-colors
                  ${selected.has(doc.id) ? 'bg-blue-50/30' : ''}`}>
                <td className="px-4 py-3" onClick={e => e.stopPropagation()}>
                  <input
                    type="checkbox"
                    className="accent-blue-600 cursor-pointer"
                    checked={selected.has(doc.id)}
                    onChange={() => toggleSelect(doc.id)}
                  />
                </td>
                <td className="px-4 py-3 text-slate-700 max-w-xs truncate">{doc.title}</td>
                <td className="px-4 py-3">
                  <span className="px-2 py-0.5 rounded text-xs bg-blue-50 text-blue-700">
                    {TYPE_LABELS[doc.type] || doc.type}
                  </span>
                </td>
                <td className="px-4 py-3">
                  <span className={`text-xs ${doc.content_length < 100 ? 'text-red-500 font-medium' : doc.content_length < 500 ? 'text-amber-600' : 'text-slate-500'}`}>
                    {doc.content_length?.toLocaleString() ?? 0}
                  </span>
                </td>
                <td className="px-4 py-3">
                  <span className={`px-2 py-0.5 rounded text-xs ${STATUS_COLORS[doc.status] || 'bg-slate-100 text-slate-600'}`}>
                    {doc.status === 'extracted' ? '已抽取' : '待处理'}
                  </span>
                </td>
                <td className="px-4 py-3 text-slate-400">{doc.created_at?.slice(0, 10)}</td>
                <td className="px-4 py-3" onClick={e => e.stopPropagation()}>
                  <div className="flex items-center justify-end gap-2">
                    <button
                      onClick={() => setViewingId(doc.id)}
                      className="text-xs text-blue-600 hover:underline"
                    >查看</button>
                    <button
                      onClick={() => askDeleteOne(doc)}
                      className="text-xs text-slate-400 hover:text-red-500 p-1 rounded hover:bg-red-50"
                      title="删除该文档"
                    ><Trash2 size={12} /></button>
                  </div>
                </td>
              </tr>
            ))}
            {!loading && data.items.length === 0 && (
              <tr><td colSpan={7} className="px-4 py-8 text-center text-slate-400">
                {params.min_content_length > 0
                  ? `当前视图筛选下无文档（≥${params.min_content_length} 字符）`
                  : '暂无数据，请先点击「加载数据文件」'}
              </td></tr>
            )}
          </tbody>
        </table>

        <div className="p-4 flex items-center justify-between text-sm text-slate-500">
          <span>第 {params.page} 页，共 {Math.ceil(data.total / params.page_size) || 1} 页</span>
          <div className="flex gap-2">
            <button
              disabled={params.page <= 1}
              onClick={() => { const p = { ...params, page: params.page - 1 }; setParams(p); fetchList(p) }}
              className="px-3 py-1 border border-slate-200 rounded disabled:opacity-40 hover:bg-slate-50"
            >上一页</button>
            <button
              disabled={params.page * params.page_size >= data.total}
              onClick={() => { const p = { ...params, page: params.page + 1 }; setParams(p); fetchList(p) }}
              className="px-3 py-1 border border-slate-200 rounded disabled:opacity-40 hover:bg-slate-50"
            >下一页</button>
          </div>
        </div>
      </div>

      {viewingId && <DocumentViewer docId={viewingId} onClose={() => setViewingId(null)} />}
      {confirm && (
        <ConfirmDialog
          title={confirm.title}
          message={confirm.message}
          detail={confirm.detail}
          confirmLabel={confirm.confirmLabel}
          tone={confirm.tone}
          loading={confirmLoading}
          onConfirm={runConfirm}
          onCancel={() => !confirmLoading && setConfirm(null)}
        />
      )}
    </div>
  )
}
