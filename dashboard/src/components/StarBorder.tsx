import React from 'react'

type StarBorderProps<T extends React.ElementType = 'button'> = {
  as?: T
  className?: string
  color?: string
  speed?: string
  children?: React.ReactNode
} & React.ComponentPropsWithoutRef<T>

const StarBorder = <T extends React.ElementType = 'button'>({
  as,
  className = '',
  color = '#e4587a',
  speed = '6s',
  children,
  ...rest
}: StarBorderProps<T>) => {
  const Component = (as || 'button') as React.ElementType

  return (
    <Component className={`star-border-container ${className}`} {...rest}>
      <div
        className="star-border-gradient star-border-gradient--top"
        style={{
          background: `radial-gradient(circle, ${color}, transparent 10%)`,
          animationDuration: speed,
        }}
      />
      <div
        className="star-border-gradient star-border-gradient--bottom"
        style={{
          background: `radial-gradient(circle, ${color}, transparent 10%)`,
          animationDuration: speed,
        }}
      />
      <div className="star-border-inner">{children}</div>
    </Component>
  )
}

export default StarBorder
