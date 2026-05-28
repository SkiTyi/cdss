import { useEffect, useState } from 'react'
import { globalStats, documents, extraction, training } from '../api/client'
import { FileText, Zap, Database, FlaskConical, TrendingUp, RefreshCw } from 'lucide-react'

function StatCard({ icon: Icon, label, value, color }) {
  return (
    <div className="bg-white rounded-xl p-5 shadow-sm border border-slate-100">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm text-slate-500">{label}</p>
          <p className="text-2xl font-bold text-slate-800 mt-1">{value ?? '—'}</p>
        </div>
        <div className={`p-3 rounded-lg ${color}`}>
          <Icon size={20} className="text-white" />
        </div>
      </div>
    </div>
  )
}

export default function Dashboard() {
  const [stats, setStats] = useState(null)
  const [docStats, setDocStats] = useState(null)
  const [extStats, setExtStats] = useState(null)
  const [trainStats, setTrainStats] = useState(null)
  const [loading, setLoading] = useState(true)

  const load = async () => {
    setLoading(true)
    try {
      const [s, d, e, t] = await Promise.all([
        globalStats(), documents.stats(), extraction.stats(), training.stats()
      ])
      setStats(s.data)
      setDocStats(d.data)
      setExtStats(e.data)
      setTrainStats(t.data)
    } catch {}
    setLoading(false)
  }

  useEffect(() => { load() }, [])

  return (
    <div className="p-8">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-xl font-bold text-slate-800">总览</h1>
          <p className="text-sm text-slate-500 mt-0.5">医学知识蒸馏平台运行状态</p>
        </div>
        <button onClick={load} className="flex items-center gap-2 text-sm text-slate-500 hover:text-slate-700">
          <RefreshCw size={14} className={loading ? 'animate-spin' : ''} /> 刷新
        </button>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        <StatCard icon={FileText} label="文档总数" value={stats?.documents} color="bg-blue-500" />
        <StatCard icon={Zap} label="诊断实例" value={stats?.diagnostic_instances} color="bg-purple-500" />
        <StatCard icon={Database} label="数据集" value={stats?.datasets} color="bg-green-500" />
        <StatCard icon={FlaskConical} label="训练实验" value={stats?.experiments} color="bg-orange-500" />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="bg-white rounded-xl p-5 shadow-sm border border-slate-100">
          <h2 className="text-sm font-semibold text-slate-700 mb-3">文档来源</h2>
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-slate-500">临床病例报告</span>
              <span className="font-medium">{docStats?.case_reports ?? 0}</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-slate-500">临床指南/共识</span>
              <span className="font-medium">{docStats?.guidelines ?? 0}</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-slate-500">已完成抽取</span>
              <span className="font-medium text-green-600">{docStats?.extracted ?? 0}</span>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-xl p-5 shadow-sm border border-slate-100">
          <h2 className="text-sm font-semibold text-slate-700 mb-3">知识抽取</h2>
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-slate-500">抽取任务总数</span>
              <span className="font-medium">{extStats?.total_jobs ?? 0}</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-slate-500">运行中</span>
              <span className="font-medium text-blue-600">{extStats?.running_jobs ?? 0}</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-slate-500">已审核条目</span>
              <span className="font-medium text-green-600">{extStats?.approved_items ?? 0}</span>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-xl p-5 shadow-sm border border-slate-100">
          <h2 className="text-sm font-semibold text-slate-700 mb-3">模型训练</h2>
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-slate-500">实验总数</span>
              <span className="font-medium">{trainStats?.total ?? 0}</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-slate-500">训练中</span>
              <span className="font-medium text-blue-600">{trainStats?.running ?? 0}</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-slate-500">已完成</span>
              <span className="font-medium text-green-600">{trainStats?.completed ?? 0}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
