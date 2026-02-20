import { useBackgroundTasks } from '../contexts/BackgroundTaskContext'
import { CheckCircle2, AlertCircle, Loader } from 'lucide-react'
import './BackgroundTaskIndicator.css'

export function BackgroundTaskIndicator() {
  const { tasks } = useBackgroundTasks()

  if (tasks.length === 0) {
    return null
  }

  // Group tasks by bot and type
  const groupedTasks = tasks.reduce(
    (acc, task) => {
      const key = `${task.botId}-${task.type}`
      if (!acc[key]) {
        acc[key] = task
      }
      return acc
    },
    {} as Record<string, typeof tasks[0]>
  )

  const taskList = Object.values(groupedTasks)

  return (
    <div className="bg-task-indicator">
      <div className="bg-task-container">
        {taskList.map((task) => (
          <div key={task.id} className={`bg-task-item bg-task-${task.status}`}>
            <div className="bg-task-icon">
              {task.status === 'in_progress' && <Loader size={16} className="animate-spin" />}
              {task.status === 'done' && <CheckCircle2 size={16} />}
              {task.status === 'error' && <AlertCircle size={16} />}
              {task.status === 'pending' && <Loader size={16} className="animate-spin" />}
            </div>
            <div className="bg-task-text">
              <div className="bg-task-label">Generating persona</div>
              {task.message && <div className="bg-task-message">{task.message}</div>}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
