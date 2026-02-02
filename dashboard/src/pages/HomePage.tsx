import { Link } from 'react-router-dom'

export default function HomePage() {
  return (
    <div className="page">
      <nav className="home-nav">
        <Link to="/dashboard" className="home-nav-link">
          Dashboard
        </Link>
      </nav>
      <div className="home-content" />
    </div>
  )
}
