// Minimal, dependency-free markdown renderer.
// Handles: headings (#…######), fenced code blocks (```), unordered/ordered
// lists, blockquotes, hr, inline `code`, **bold**, *italic*, [text](url),
// images ![alt](url) and paragraph wrapping. Good enough for previewing
// guideline / case-report MD files without pulling react-markdown.

function escapeHtml(s) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
}

function renderInline(text) {
  let s = escapeHtml(text)
  // inline code first to avoid clobbering * inside `code`
  s = s.replace(/`([^`]+?)`/g, '<code class="px-1 py-0.5 bg-slate-100 text-pink-600 rounded text-[0.85em]">$1</code>')
  // images
  s = s.replace(/!\[([^\]]*)\]\(([^)\s]+)\)/g,
    '<img alt="$1" src="$2" class="inline-block max-w-full my-2 rounded" />')
  // links
  s = s.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g,
    '<a href="$2" target="_blank" rel="noreferrer" class="text-blue-600 hover:underline">$1</a>')
  // bold
  s = s.replace(/\*\*([^*]+?)\*\*/g, '<strong class="font-semibold text-slate-800">$1</strong>')
  s = s.replace(/__([^_]+?)__/g, '<strong class="font-semibold text-slate-800">$1</strong>')
  // italic (single * or _, but not adjacent to word chars on the wrong side)
  s = s.replace(/(^|[\s(])\*([^*\n]+?)\*(?=[\s).,!?]|$)/g, '$1<em>$2</em>')
  s = s.replace(/(^|[\s(])_([^_\n]+?)_(?=[\s).,!?]|$)/g, '$1<em>$2</em>')
  return s
}

function renderMarkdown(md) {
  if (!md) return ''
  const lines = md.replace(/\r\n/g, '\n').split('\n')
  const out = []
  let i = 0
  let listType = null  // 'ul' | 'ol' | null
  let paraBuf = []

  const flushPara = () => {
    if (paraBuf.length) {
      out.push(`<p class="my-2 leading-relaxed">${renderInline(paraBuf.join(' '))}</p>`)
      paraBuf = []
    }
  }
  const closeList = () => {
    if (listType) {
      out.push(`</${listType}>`)
      listType = null
    }
  }

  while (i < lines.length) {
    const line = lines[i]

    // fenced code block
    const fence = line.match(/^```(\w+)?\s*$/)
    if (fence) {
      flushPara(); closeList()
      const lang = fence[1] || ''
      const code = []
      i++
      while (i < lines.length && !/^```\s*$/.test(lines[i])) {
        code.push(lines[i]); i++
      }
      i++ // consume closing fence
      out.push(
        `<pre class="my-3 p-3 bg-slate-900 text-slate-100 rounded-lg overflow-x-auto text-xs"><code data-lang="${lang}">${escapeHtml(code.join('\n'))}</code></pre>`,
      )
      continue
    }

    // blank line
    if (/^\s*$/.test(line)) { flushPara(); closeList(); i++; continue }

    // heading
    const h = line.match(/^(#{1,6})\s+(.+?)\s*#*\s*$/)
    if (h) {
      flushPara(); closeList()
      const level = h[1].length
      const sizes = ['text-2xl', 'text-xl', 'text-lg', 'text-base', 'text-sm', 'text-xs']
      const cls = `${sizes[level - 1]} font-semibold text-slate-800 mt-4 mb-2`
      out.push(`<h${level} class="${cls}">${renderInline(h[2])}</h${level}>`)
      i++; continue
    }

    // hr
    if (/^\s*(\*\s*\*\s*\*+|-{3,}|_{3,})\s*$/.test(line)) {
      flushPara(); closeList()
      out.push('<hr class="my-4 border-slate-200" />')
      i++; continue
    }

    // unordered list
    const ul = line.match(/^\s*[-*+]\s+(.+)$/)
    if (ul) {
      flushPara()
      if (listType !== 'ul') { closeList(); out.push('<ul class="list-disc list-outside ml-6 my-2 space-y-1">'); listType = 'ul' }
      out.push(`<li>${renderInline(ul[1])}</li>`)
      i++; continue
    }
    // ordered list
    const ol = line.match(/^\s*\d+[.)]\s+(.+)$/)
    if (ol) {
      flushPara()
      if (listType !== 'ol') { closeList(); out.push('<ol class="list-decimal list-outside ml-6 my-2 space-y-1">'); listType = 'ol' }
      out.push(`<li>${renderInline(ol[1])}</li>`)
      i++; continue
    }

    // blockquote
    const bq = line.match(/^\s*>\s?(.*)$/)
    if (bq) {
      flushPara(); closeList()
      out.push(`<blockquote class="my-2 pl-3 border-l-4 border-slate-300 text-slate-600 italic">${renderInline(bq[1])}</blockquote>`)
      i++; continue
    }

    // paragraph accumulation
    closeList()
    paraBuf.push(line.trim())
    i++
  }
  flushPara(); closeList()
  return out.join('\n')
}

export default function MarkdownView({ source, className = '' }) {
  const html = renderMarkdown(source || '')
  return (
    <div
      className={`text-sm text-slate-700 ${className}`}
      // Source comes from our own API and is plain MD; we escape HTML inside renderInline.
      dangerouslySetInnerHTML={{ __html: html }}
    />
  )
}
