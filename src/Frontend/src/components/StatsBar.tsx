import type { ReactNode } from 'react';
import { Zap, AlertTriangle, BatteryWarning, Users } from 'lucide-react';
import { cn } from '../utils';

interface StatCardProps {
  title: string;
  value: string | number;
  icon: ReactNode;
  trend?: string;
  trendUp?: boolean;
  alert?: boolean;
}

function StatCard({ title, value, icon, trend, trendUp, alert }: StatCardProps) {
  return (
    <div className="card p-5 flex flex-col gap-4">
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium text-textMuted">{title}</span>
        <div className={cn("p-2 rounded-lg bg-slate-100", alert ? "text-danger" : "text-primary")}>
          {icon}
        </div>
      </div>
      <div className="flex items-end justify-between">
        <span className="text-2xl font-bold">{value}</span>
        {trend && (
          <span className={cn("text-xs font-medium", trendUp ? "text-danger" : "text-success")}>
            {trend}
          </span>
        )}
      </div>
    </div>
  );
}

export function StatsBar() {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
      <StatCard 
        title="Total Substation Assets" 
        value="1,245" 
        icon={<Zap className="w-5 h-5" />} 
        trend="+12 this month"
        trendUp={false}
      />
      <StatCard 
        title="Active Critical Alerts" 
        value="4" 
        icon={<AlertTriangle className="w-5 h-5" />} 
        alert 
        trend="+2 since yesterday"
        trendUp={true}
      />
      <StatCard 
        title="At-Risk Grid Capacity" 
        value="12%" 
        icon={<BatteryWarning className="w-5 h-5" />} 
        trend="High Stress Level"
        trendUp={true}
      />
      <StatCard 
        title="Pre-positioned Crews" 
        value="8 / 12" 
        icon={<Users className="w-5 h-5" />} 
        trend="4 available"
        trendUp={false}
      />
    </div>
  );
}
