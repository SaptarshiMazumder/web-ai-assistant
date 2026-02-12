import { useCallback, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { FileText, Upload, Plus, X, FileIcon } from 'lucide-react'
import { SegmentedTabs, UiButton, type SegmentedTabOption } from '../../components/ui'
import { useCreateBotFlow } from './CreateBotContext'
import { FileDropzone } from '../../components/FileDropzone'

type TabId = 'pdfs' | 'text-docs' | 'drive' | 'custom-text'

export default function CreateBotAdditionalSourcesPage() {
  const navigate = useNavigate()
  const { flow, step2 } = useCreateBotFlow()
  const { pdfFiles, setPdfFiles, textDocFiles, setTextDocFiles, customTextEntries, setCustomTextEntries } = step2
  
  const [activeTab, setActiveTab] = useState<TabId>('pdfs')

  const hasAnySources = 
    pdfFiles.length > 0 || 
    textDocFiles.length > 0 || 
    customTextEntries.some(f => f.title.trim() || f.content.trim())

  const handleAddTextField = () => {
    setCustomTextEntries([
      ...customTextEntries,
      { id: Date.now().toString(), title: '', content: '' }
    ])
  }

  const handleRemoveTextField = (id: string) => {
    if (customTextEntries.length > 1) {
      setCustomTextEntries(customTextEntries.filter(f => f.id !== id))
    }
  }

  const handleUpdateTextField = (id: string, field: 'title' | 'content', value: string) => {
    setCustomTextEntries(
      customTextEntries.map(f => f.id === id ? { ...f, [field]: value } : f)
    )
  }

  const handleContinue = useCallback(() => {
    // TODO: Save additional sources to context
    if (flow.nextPath) {
      navigate(flow.nextPath)
    }
  }, [flow.nextPath, navigate])

  const handleSkip = useCallback(() => {
    if (flow.nextPath) {
      navigate(flow.nextPath)
    }
  }, [flow.nextPath, navigate])

  const tabs: SegmentedTabOption<TabId>[] = [
    { id: 'pdfs', label: 'PDF Sources', icon: <FileText size={16} /> },
    { id: 'text-docs', label: 'Text Docs', icon: <FileIcon size={16} /> },
    { id: 'drive', label: 'Drive', icon: <Upload size={16} /> },
    { id: 'custom-text', label: 'Custom Text', icon: <Plus size={16} /> },
  ]

  return (
    <div className="flow-panel-body">
      <div>
        <div className="card-title">Add additional sources (optional)</div>
        <div className="card-subtitle">
          Upload more files or add custom content to expand your assistant's knowledge.
        </div>
      </div>

      <div style={{ marginBottom: '1.5rem' }}>
        <SegmentedTabs
          value={activeTab}
          onChange={setActiveTab}
          options={tabs}
          ariaLabel="Additional sources tabs"
        />
      </div>

      {/* Tab Content */}
      <div style={{ minHeight: '300px' }}>
        {activeTab === 'pdfs' && (
          <div>
            <div style={{ marginBottom: '1rem' }}>
              <div style={{ fontWeight: 600, fontSize: '1rem', marginBottom: '0.5rem' }}>
                Upload Additional PDFs
              </div>
              <div style={{ fontSize: '0.875rem', color: 'var(--flow-muted)', marginBottom: '1rem' }}>
                Add more PDF documents to expand your assistant's knowledge base.
              </div>
            </div>
            <FileDropzone
              label="📄 Drop PDF files here"
              helperText="Upload additional PDFs (up to 20 files)"
              files={pdfFiles}
              setFiles={setPdfFiles}
              accept="application/pdf"
              multiple
              maxFiles={20}
            />
          </div>
        )}

        {activeTab === 'text-docs' && (
          <div>
            <div style={{ marginBottom: '1rem' }}>
              <div style={{ fontWeight: 600, fontSize: '1rem', marginBottom: '0.5rem' }}>
                Upload Text Documents
              </div>
              <div style={{ fontSize: '0.875rem', color: 'var(--flow-muted)', marginBottom: '1rem' }}>
                Upload .txt, .md, or other text files for your assistant to learn from.
              </div>
            </div>
            <FileDropzone
              label="📝 Drop text files here"
              helperText="Upload .txt, .md, .doc, .docx files (up to 20 files)"
              files={textDocFiles}
              setFiles={setTextDocFiles}
              accept=".txt,.md,.doc,.docx,text/plain,text/markdown,application/msword,application/vnd.openxmlformats-officedocument.wordprocessingml.document"
              multiple
              maxFiles={20}
            />
          </div>
        )}

        {activeTab === 'drive' && (
          <div style={{ 
            textAlign: 'center', 
            padding: '4rem 2rem',
            background: 'var(--flow-surface)',
            borderRadius: 'var(--flow-radius)',
            border: '1px dashed var(--flow-border)'
          }}>
            <Upload size={48} color="var(--flow-muted)" style={{ marginBottom: '1rem' }} />
            <div style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--flow-heading)', marginBottom: '0.5rem' }}>
              Google Drive Integration
            </div>
            <div style={{ fontSize: '0.95rem', color: 'var(--flow-muted)', marginBottom: '1.5rem' }}>
              Coming soon! You'll be able to sync files directly from your Google Drive.
            </div>
            <div style={{ 
              display: 'inline-block',
              padding: '0.5rem 1rem',
              background: 'var(--flow-accent)',
              color: 'white',
              borderRadius: '6px',
              fontSize: '0.8rem',
              fontWeight: 600,
              letterSpacing: '0.05em'
            }}>
              COMING SOON
            </div>
          </div>
        )}

        {activeTab === 'custom-text' && (
          <div>
            <div style={{ marginBottom: '1rem' }}>
              <div style={{ fontWeight: 600, fontSize: '1rem', marginBottom: '0.5rem' }}>
                Add Custom Text
              </div>
              <div style={{ fontSize: '0.875rem', color: 'var(--flow-muted)', marginBottom: '1rem' }}>
                Create custom text entries for FAQs, policies, or any other important information.
              </div>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              {customTextEntries.map((field, index) => (
                <div 
                  key={field.id}
                  style={{ 
                    background: 'var(--flow-surface)',
                    border: '1px solid var(--flow-border)',
                    borderRadius: 'var(--flow-radius)',
                    padding: '1rem'
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
                    <div style={{ fontWeight: 600, fontSize: '0.9rem', color: 'var(--flow-heading)' }}>
                      Entry #{index + 1}
                    </div>
                    {customTextEntries.length > 1 && (
                      <button
                        onClick={() => handleRemoveTextField(field.id)}
                        style={{
                          background: 'transparent',
                          border: 'none',
                          cursor: 'pointer',
                          padding: '0.25rem',
                          color: 'var(--flow-muted)',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center'
                        }}
                        title="Remove this entry"
                      >
                        <X size={18} />
                      </button>
                    )}
                  </div>
                  
                  <div style={{ marginBottom: '0.75rem' }}>
                    <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, marginBottom: '0.5rem', color: 'var(--flow-heading)' }}>
                      Title
                    </label>
                    <input
                      type="text"
                      value={field.title}
                      onChange={(e) => handleUpdateTextField(field.id, 'title', e.target.value)}
                      placeholder="e.g., Return Policy, Office Hours, etc."
                      style={{ width: '100%' }}
                    />
                  </div>

                  <div>
                    <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, marginBottom: '0.5rem', color: 'var(--flow-heading)' }}>
                      Content
                    </label>
                    <textarea
                      value={field.content}
                      onChange={(e) => handleUpdateTextField(field.id, 'content', e.target.value)}
                      placeholder="Enter the full text content here..."
                      rows={6}
                      style={{ 
                        width: '100%',
                        fontFamily: 'inherit',
                        resize: 'vertical'
                      }}
                    />
                  </div>
                </div>
              ))}
            </div>

            <UiButton
              variant="secondary"
              onClick={handleAddTextField}
              style={{ marginTop: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}
            >
              <Plus size={16} />
              Add Another Entry
            </UiButton>
          </div>
        )}
      </div>

      <div className="flow-actions" style={{ marginTop: '2rem' }}>
        <UiButton variant="secondary" onClick={() => flow.prevPath && navigate(flow.prevPath)}>
          Back
        </UiButton>
        <div style={{ display: 'flex', gap: '0.75rem', marginLeft: 'auto' }}>
          <UiButton variant="ghost" onClick={handleSkip}>
            Skip for now
          </UiButton>
          <UiButton variant="primary" onClick={handleContinue} disabled={!hasAnySources}>
            Continue
          </UiButton>
        </div>
      </div>
    </div>
  )
}
