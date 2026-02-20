import React, { createContext, useState, useCallback } from 'react'

export type BackgroundTask = {
  id: string
  botId: string
  type: 'persona'
  status: 'pending' | 'in_progress' | 'done' | 'error'
  message?: string
  createdAt: number
}

type BackgroundTaskContextType = {
  tasks: BackgroundTask[]
  addTask: (botId: string, type: BackgroundTask['type'], message?: string) => string
  updateTask: (id: string, status: BackgroundTask['status'], message?: string) => void
  removeTask: (id: string) => void
  hasActiveTasks: boolean
}

export const BackgroundTaskContext = createContext<BackgroundTaskContextType | undefined>(undefined)

export function BackgroundTaskProvider({ children }: { children: React.ReactNode }) {
  const [tasks, setTasks] = useState<BackgroundTask[]>([])

  const addTask = useCallback((botId: string, type: BackgroundTask['type'], message?: string) => {
    const id = `${botId}-${type}-${Date.now()}`
    const newTask: BackgroundTask = {
      id,
      botId,
      type,
      status: 'in_progress',
      message,
      createdAt: Date.now(),
    }
    setTasks((prev) => [...prev, newTask])
    return id
  }, [])

  const updateTask = useCallback((id: string, status: BackgroundTask['status'], message?: string) => {
    setTasks((prev) =>
      prev.map((task) =>
        task.id === id ? { ...task, status, message: message || task.message } : task
      )
    )
  }, [])

  const removeTask = useCallback((id: string) => {
    setTasks((prev) => prev.filter((task) => task.id !== id))
  }, [])

  const hasActiveTasks = tasks.some((t) => t.status === 'in_progress' || t.status === 'pending')

  return (
    <BackgroundTaskContext.Provider value={{ tasks, addTask, updateTask, removeTask, hasActiveTasks }}>
      {children}
    </BackgroundTaskContext.Provider>
  )
}

export function useBackgroundTasks() {
  const context = React.useContext(BackgroundTaskContext)
  if (!context) {
    throw new Error('useBackgroundTasks must be used within BackgroundTaskProvider')
  }
  return context
}
