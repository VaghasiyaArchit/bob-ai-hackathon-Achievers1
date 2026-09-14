
import { Activity, Bell, Settings, User } from 'lucide-react';

export function TopNav() {
  return (
    <header className="flex items-center justify-between px-6 py-4 bg-surface border-b border-border">
      <div className="flex items-center gap-3">
        <div className="p-2 bg-primary/10 rounded-lg">
          <Activity className="w-6 h-6 text-primary" />
        </div>
        <h1 className="text-xl font-bold tracking-tight text-text">GridPulse <span className="text-primary">AI</span></h1>
      </div>
      <div className="flex items-center gap-4">
        <button className="p-2 text-textMuted hover:text-text transition-colors relative">
          <Bell className="w-5 h-5" />
          <span className="absolute top-1.5 right-1.5 w-2 h-2 bg-danger rounded-full"></span>
        </button>
        <button className="p-2 text-textMuted hover:text-text transition-colors">
          <Settings className="w-5 h-5" />
        </button>
        <div className="w-8 h-8 rounded-full bg-slate-100 flex items-center justify-center border border-slate-200">
          <User className="w-4 h-4 text-slate-600" />
        </div>
      </div>
    </header>
  );
}
