import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import Documents from './pages/Documents'
import Extraction from './pages/Extraction'
import Datasets from './pages/Datasets'
import Training from './pages/Training'
import Assistant from './pages/Assistant'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<Dashboard />} />
          <Route path="/documents" element={<Documents />} />
          <Route path="/extraction" element={<Extraction />} />
          <Route path="/datasets" element={<Datasets />} />
          <Route path="/training" element={<Training />} />
          <Route path="/assistant" element={<Assistant />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
