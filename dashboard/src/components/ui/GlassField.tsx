import React, { type ReactNode } from 'react'

type GlassFieldProps = {
    label: string
    helper?: string
    error?: string
    children: ReactNode
    className?: string
    style?: React.CSSProperties
}

export function GlassField({
    label,
    helper,
    error,
    children,
    className = '',
    style,
}: GlassFieldProps) {
    return (
        <div className={`ui-glass-field ${className}`} style={style}>
            <label className="ui-glass-field-label">{label}</label>
            <div className="ui-glass-field-input-wrap">
                {children}
            </div>
            {(error || helper) && (
                <span className={`ui-glass-field-helper ${error ? 'error' : ''}`}>
                    {error || helper}
                </span>
            )}
        </div>
    )
}
