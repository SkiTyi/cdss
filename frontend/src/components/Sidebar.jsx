import { NavLink } from 'react-router-dom'
import {
  LayoutDashboard, FileText, Zap, Database, FlaskConical, Stethoscope
} from 'lucide-react'

const links = [
  { to: '/', icon: LayoutDashboard, label: '总览' },
  { to: '/documents', icon: FileText, label: '文档管理' },
  { to: '/extraction', icon: Zap, label: '知识抽取' },
  { to: '/datasets', icon: Database, label: '数据集' },
  { to: '/training', icon: FlaskConical, label: '训练监控' },
  { to: '/assistant', icon: Stethoscope, label: '临床助手' },
]

export default function Sidebar() {
  return (
    <aside className="w-56 bg-slate-900 text-slate-100 flex flex-col min-h-screen shrink-0">
      <div className="px-5 py-5 border-b border-slate-700">
        <div className="text-base font-semibold text-white leading-tight">CDSS</div>
        <div className="text-xs text-slate-400 mt-0.5">医学知识蒸馏平台</div>
      </div>
      <nav className="flex-1 py-4 space-y-0.5 px-2">
        {links.map(({ to, icon: Icon, label }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className={({ isActive }) =>
              `flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-colors ${
                isActive
                  ? 'bg-blue-600 text-white'
                  : 'text-slate-300 hover:bg-slate-800 hover:text-white'
              }`
            }
          >
            <Icon size={16} />
            {label}
          </NavLink>
        ))}
      </nav>
    </aside>
  )
}
