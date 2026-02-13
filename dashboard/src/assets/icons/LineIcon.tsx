import { forwardRef } from 'react'
import type { LucideProps } from 'lucide-react'

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const LineIcon = forwardRef<any, LucideProps>(({
    size = 24,
    color,
    strokeWidth,
    className,
    style,
    absoluteStrokeWidth,
    ...props
}, ref) => {
    const url = "https://img.icons8.com/forma-light/96/line-me.png"

    return (
        <span
            ref={ref}
            style={{
                width: size,
                height: size,
                display: 'inline-block',
                backgroundColor: 'currentColor',
                maskImage: `url('${url}')`,
                maskSize: 'contain',
                maskRepeat: 'no-repeat',
                maskPosition: 'center',
                WebkitMaskImage: `url('${url}')`,
                WebkitMaskSize: 'contain',
                WebkitMaskRepeat: 'no-repeat',
                WebkitMaskPosition: 'center',
                ...style
            }}
            className={className}
            role="img"
            aria-label="LINE"
            {...(props as any)}
        />
    )
})
