import { useTranslation } from 'react-i18next'
import { Check, Globe } from 'lucide-react'
import { useState, useRef, useEffect } from 'react'

const LANGUAGES = [
    { code: 'en', label: 'English' },
    { code: 'ja', label: '日本語' }
    // Add more languages here as they become available
]

export function LanguageSelector() {
    const { i18n } = useTranslation()
    const [isOpen, setIsOpen] = useState(false)
    const dropdownRef = useRef<HTMLDivElement>(null)

    const currentLangCode = i18n.language || 'en'
    const currentLang = LANGUAGES.find(l => l.code === currentLangCode) || LANGUAGES[0]

    const toggleDropdown = () => setIsOpen(!isOpen)

    const selectLanguage = (code: string) => {
        i18n.changeLanguage(code)
        setIsOpen(false)
    }

    // Close dropdown when clicking outside
    useEffect(() => {
        const handleClickOutside = (event: MouseEvent) => {
            if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
                setIsOpen(false)
            }
        }
        document.addEventListener('mousedown', handleClickOutside)
        return () => document.removeEventListener('mousedown', handleClickOutside)
    }, [])

    return (
        <div className="language-selector" ref={dropdownRef} style={{ position: 'relative' }}>
            <button
                onClick={toggleDropdown}
                style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px',
                    padding: '8px 16px',
                    background: 'var(--flow-surface)',
                    border: '1px solid var(--flow-border)',
                    borderRadius: 'var(--flow-radius)',
                    color: 'var(--flow-text)',
                    cursor: 'pointer',
                    fontWeight: 500,
                    userSelect: 'none'
                }}
            >
                <Globe size={18} color="var(--flow-muted)" />
                {currentLang.label}
            </button>

            {isOpen && (
                <div
                    style={{
                        position: 'absolute',
                        top: '100%',
                        left: 0,
                        marginTop: '8px',
                        background: 'var(--flow-bg)',
                        border: '1px solid var(--flow-border)',
                        borderRadius: 'var(--flow-radius)',
                        boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
                        minWidth: '200px',
                        zIndex: 100,
                        overflow: 'hidden'
                    }}
                >
                    {LANGUAGES.map((lang) => (
                        <div
                            key={lang.code}
                            onClick={() => selectLanguage(lang.code)}
                            style={{
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'space-between',
                                padding: '10px 16px',
                                cursor: 'pointer',
                                background: currentLangCode === lang.code ? 'var(--flow-surface)' : 'transparent',
                                color: currentLangCode === lang.code ? 'var(--flow-accent)' : 'var(--flow-text)'
                            }}
                            onMouseEnter={(e) => {
                                e.currentTarget.style.background = 'var(--flow-surface)'
                            }}
                            onMouseLeave={(e) => {
                                e.currentTarget.style.background = currentLangCode === lang.code ? 'var(--flow-surface)' : 'transparent'
                            }}
                        >
                            <span>{lang.label}</span>
                            {currentLangCode === lang.code && <Check size={16} />}
                        </div>
                    ))}
                </div>
            )}
        </div>
    )
}
