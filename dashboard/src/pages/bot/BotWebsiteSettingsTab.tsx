import { useState } from 'react'
import { Code2, Copy, Check, Sparkles, Globe, Rocket } from 'lucide-react'
import { AnimatedPage, SectionHeader, GlassCard, UiButton } from '../../components/ui'
import { useParams } from 'react-router-dom'
import { useDashboardData } from '../../hooks/useDashboardData'

export default function BotWebsiteSettingsTab() {
    const { botId } = useParams()
    const { embedSnippet, copySnippet } = useDashboardData()
    const [copied, setCopied] = useState(false)

    const handleCopy = () => {
        void copySnippet()
        setCopied(true)
        setTimeout(() => setCopied(false), 2500)
    }

    if (!botId) {
        return <div className="empty-panel">Select a bot to view website installation.</div>
    }

    return (
        <AnimatedPage>
            <SectionHeader
                eyebrow="Install"
                title="Install on Website"
                subtitle="Deploy your AI agent to any website with a single code snippet."
            />

            {/* Hero Card */}
            <div style={{
                background: 'linear-gradient(135deg, #f6b46d 0%, #f0806b 48%, #e66397 100%)',
                borderRadius: '20px',
                padding: '2.5rem',
                marginBottom: '2rem',
                position: 'relative',
                overflow: 'hidden',
                boxShadow: '0 20px 60px rgba(228, 88, 122, 0.3)',
            }}>
                <div style={{
                    position: 'absolute',
                    top: '-50%',
                    right: '-10%',
                    width: '400px',
                    height: '400px',
                    background: 'radial-gradient(circle, rgba(255,255,255,0.15) 0%, transparent 70%)',
                    borderRadius: '50%',
                    pointerEvents: 'none',
                }} />

                <div style={{ position: 'relative', zIndex: 1 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '1rem' }}>
                        <div style={{
                            width: '48px',
                            height: '48px',
                            borderRadius: '14px',
                            background: 'rgba(255,255,255,0.2)',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            backdropFilter: 'blur(10px)',
                        }}>
                            <Sparkles size={24} color="#fff" />
                        </div>
                        <h2 style={{
                            fontSize: '1.75rem',
                            fontWeight: 700,
                            color: '#fff',
                            margin: 0,
                        }}>
                            Ready to launch
                        </h2>
                    </div>
                    <p style={{
                        color: 'rgba(255,255,255,0.9)',
                        fontSize: '1.05rem',
                        lineHeight: '1.7',
                        margin: '0 0 1.5rem 0',
                        maxWidth: '600px',
                    }}>
                        Copy the embed code below and paste it before the closing <code style={{
                            background: 'rgba(0,0,0,0.2)',
                            padding: '2px 8px',
                            borderRadius: '4px',
                            color: '#fff',
                            fontFamily: 'monospace',
                        }}>&lt;/body&gt;</code> tag of your website. Your AI agent will appear instantly.
                    </p>
                    <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
                        <div style={{
                            display: 'flex',
                            alignItems: 'center',
                            gap: '0.5rem',
                            background: 'rgba(255,255,255,0.15)',
                            padding: '0.5rem 1rem',
                            borderRadius: '10px',
                            backdropFilter: 'blur(10px)',
                            color: '#fff',
                            fontSize: '0.9rem',
                            fontWeight: 500,
                        }}>
                            <Globe size={16} />
                            Works on any site
                        </div>
                        <div style={{
                            display: 'flex',
                            alignItems: 'center',
                            gap: '0.5rem',
                            background: 'rgba(255,255,255,0.15)',
                            padding: '0.5rem 1rem',
                            borderRadius: '10px',
                            backdropFilter: 'blur(10px)',
                            color: '#fff',
                            fontSize: '0.9rem',
                            fontWeight: 500,
                        }}>
                            <Rocket size={16} />
                            Live in 30 seconds
                        </div>
                    </div>
                </div>
            </div>

            {/* Code Snippet Card */}
            <GlassCard style={{ marginBottom: '2rem' }}>
                <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.75rem',
                    marginBottom: '1.25rem',
                }}>
                    <div style={{
                        width: '36px',
                        height: '36px',
                        borderRadius: '10px',
                        background: 'linear-gradient(135deg, #f6b46d 0%, #f0806b 48%, #e66397 100%)',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                    }}>
                        <Code2 size={18} color="#fff" />
                    </div>
                    <h3 style={{
                        fontSize: '1.25rem',
                        fontWeight: 600,
                        margin: 0,
                    }}>
                        Your embed code
                    </h3>
                </div>

                <div style={{ display: 'grid', gap: '1rem' }}>
                    <div style={{ position: 'relative' }}>
                        <pre style={{
                            background: 'var(--ui-flow-surface)',
                            color: 'var(--ui-flow-text)',
                            padding: '1.5rem',
                            borderRadius: '12px',
                            fontSize: '0.9rem',
                            lineHeight: '1.7',
                            overflow: 'auto',
                            margin: 0,
                            border: '1.5px solid var(--ui-flow-border)',
                            fontFamily: '"Fira Code", "Consolas", monospace',
                            whiteSpace: 'pre-wrap',
                            wordBreak: 'break-all',
                            minHeight: '120px',
                        }}>
                            {embedSnippet || '// Loading...'}
                        </pre>
                    </div>

                    <div style={{ display: 'flex', justifyContent: 'flex-end', alignItems: 'center', gap: '1rem' }}>
                        <p style={{
                            color: 'var(--text-secondary)',
                            fontSize: '0.9rem',
                            margin: 0,
                            display: 'flex',
                            alignItems: 'center',
                            gap: '0.5rem',
                        }}>
                            <span style={{
                                background: 'var(--ui-flow-brand-gradient)',
                                WebkitBackgroundClip: 'text',
                                WebkitTextFillColor: 'transparent',
                                fontWeight: 600,
                            }}>
                                💡 Tip:
                            </span>
                            Paste before the &lt;/body&gt; tag.
                        </p>

                        <UiButton
                            variant={copied ? 'primary' : 'secondary'}
                            onClick={handleCopy}
                            disabled={!embedSnippet}
                            style={{ minWidth: '140px' }}
                        >
                            {copied ? <Check size={18} /> : <Copy size={18} />}
                            {copied ? 'Copied!' : 'Copy snippet'}
                        </UiButton>
                    </div>
                </div>
            </GlassCard>

            {/* Instructions Card */}
            <GlassCard>
                <h3 style={{
                    fontSize: '1.25rem',
                    fontWeight: 600,
                    marginBottom: '1.5rem',
                }}>
                    How to install
                </h3>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1.5rem' }}>
                    {[
                        { step: '1', title: 'Copy the embed code', emoji: '📋' },
                        { step: '2', title: 'Open your HTML file', emoji: '📄' },
                        { step: '3', title: 'Find the </body> tag', emoji: '🔍' },
                        { step: '4', title: 'Paste it before it', emoji: '✨' },
                    ].map((item) => (
                        <div key={item.step} style={{
                            display: 'flex',
                            flexDirection: 'column',
                            gap: '0.75rem',
                        }}>
                            <div style={{
                                width: '36px',
                                height: '36px',
                                borderRadius: '10px',
                                background: 'var(--ui-flow-brand-gradient)',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                fontSize: '1rem',
                                fontWeight: 700,
                                color: '#fff',
                                boxShadow: '0 4px 12px rgba(228, 88, 122, 0.3)',
                            }}>
                                {item.step}
                            </div>
                            <div style={{
                                fontSize: '0.95rem',
                                fontWeight: 500,
                                lineHeight: '1.4',
                            }}>
                                <span style={{ marginRight: '0.4rem' }}>{item.emoji}</span>
                                {item.title}
                            </div>
                        </div>
                    ))}
                </div>
            </GlassCard>
        </AnimatedPage >
    )
}
