import { useState } from 'react'
import { Code2, Copy, Check, Sparkles, Globe, Rocket } from 'lucide-react'
import { AnimatedPage, SectionHeader, GlassCard, UiButton } from '../../components/ui'
import { useParams } from 'react-router-dom'
import { useDashboardData } from '../../hooks/useDashboardData'
import { useTranslation } from 'react-i18next'

export default function BotWebsiteSettingsTab() {
    const { botId } = useParams()
    const { embedSnippet, copySnippet } = useDashboardData()
    const [copied, setCopied] = useState(false)
    const { i18n } = useTranslation()
    const lang = (i18n.resolvedLanguage || i18n.language || '').toLowerCase()
    const isJa = lang.startsWith('ja') || lang.startsWith('jp')
    const tr = (en: string, ja: string) => (isJa ? ja : en)

    const handleCopy = () => {
        void copySnippet()
        setCopied(true)
        setTimeout(() => setCopied(false), 2500)
    }

    if (!botId) {
        return <div className="empty-panel">{tr('Select a bot to view website installation.', 'Webサイト設置を表示するボットを選択してください。')}</div>
    }

    return (
        <AnimatedPage>
            <SectionHeader
                eyebrow={tr('Install', 'インストール')}
                title={tr('Install on Website', 'Webサイトにインストール')}
                subtitle={tr('Deploy your AI agent to any website with a single code snippet.', '1つのコードスニペットで、どのWebサイトにもAIエージェントを設置できます。')}
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
                            {tr('Ready to launch', '公開準備完了')}
                        </h2>
                    </div>
                    <p style={{
                        color: 'rgba(255,255,255,0.9)',
                        fontSize: '1.05rem',
                        lineHeight: '1.7',
                        margin: '0 0 1.5rem 0',
                        maxWidth: '600px',
                    }}>
                        {isJa ? (
                            <>
                                下の埋め込みコードをコピーし、サイトの <code style={{
                                    background: 'rgba(0,0,0,0.2)',
                                    padding: '2px 8px',
                                    borderRadius: '4px',
                                    color: '#fff',
                                    fontFamily: 'monospace',
                                }}>&lt;/body&gt;</code> タグの直前に貼り付けてください。AIエージェントがすぐに表示されます。
                            </>
                        ) : (
                            <>
                                Copy the embed code below and paste it before the closing <code style={{
                                    background: 'rgba(0,0,0,0.2)',
                                    padding: '2px 8px',
                                    borderRadius: '4px',
                                    color: '#fff',
                                    fontFamily: 'monospace',
                                }}>&lt;/body&gt;</code> tag of your website. Your AI agent will appear instantly.
                            </>
                        )}
                    </p>
                    <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
                        <div style={{
                            display: 'flex',
                            alignItems: 'center',
                            gap: '0.5rem',
                            background: 'rgba(255,255,255,0.15)',
                            padding: '0.5rem 1rem',
                            borderRadius: '10px',
                            color: '#fff',
                            backdropFilter: 'blur(10px)',
                            fontSize: '0.9rem',
                            fontWeight: 500,
                        }}>
                            <Globe size={16} />
                            {tr('Works on any site', 'あらゆるサイトで動作')}
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
                            {tr('Live in 30 seconds', '30秒で公開')}
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
                        {tr('Your embed code', '埋め込みコード')}
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
                            {embedSnippet || tr('// Loading...', '// 読み込み中...')}
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
                                {tr('Tip:', 'ヒント:')}
                            </span>
                            {tr('Paste before the </body> tag.', '</body> タグの直前に貼り付けてください。')}
                        </p>

                        <UiButton
                            variant={copied ? 'primary' : 'secondary'}
                            onClick={handleCopy}
                            disabled={!embedSnippet}
                            style={{ minWidth: '140px' }}
                        >
                            {copied ? <Check size={18} /> : <Copy size={18} />}
                            {copied ? tr('Copied!', 'コピー済み') : tr('Copy snippet', 'スニペットをコピー')}
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
                    {tr('How to install', 'インストール手順')}
                </h3>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1.5rem' }}>
                    {[
                        { step: '1', title: tr('Copy the embed code', '埋め込みコードをコピー'), emoji: '📋' },
                        { step: '2', title: tr('Open your HTML file', 'HTMLファイルを開く'), emoji: '📄' },
                        { step: '3', title: tr('Find the </body> tag', '</body> タグを探す'), emoji: '🔍' },
                        { step: '4', title: tr('Paste it before it', 'その直前に貼り付ける'), emoji: '✨' },
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
