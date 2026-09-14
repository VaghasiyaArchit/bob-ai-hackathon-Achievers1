
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts';
import { ShieldAlert, Info, TrendingUp, AlertTriangle, Wind, Thermometer, Droplets, Zap } from 'lucide-react';
import { type Asset } from '../data/mockData';

interface AssetInsightsProps {
  asset: Asset | null;
}

export function AssetInsights({ asset }: AssetInsightsProps) {
  if (!asset) {
    return (
      <div className="card h-full flex items-center justify-center p-8 bg-slate-50 border-dashed border-slate-200">
        <div className="text-center">
          <Info className="w-10 h-10 text-textMuted mx-auto mb-4 opacity-50" />
          <h3 className="text-lg font-medium text-text">Select an Asset</h3>
          <p className="text-sm text-textMuted mt-2 max-w-xs">
            Choose an asset from the table to view its model pipeline deep-dive.
          </p>
        </div>
      </div>
    );
  }

  // Determine stroke color for the anomaly gauge
  const anomalyPercentage = Math.round(asset.model1.anomalyScore * 100);
  const strokeColor = anomalyPercentage >= 80 ? '#E07A5F' : anomalyPercentage >= 40 ? '#F2CC8F' : '#81B29A';

  return (
    <div className="card h-full flex flex-col overflow-hidden">
      <div className="p-5 border-b border-border bg-slate-100 flex items-start justify-between">
        <div>
          <h2 className="text-lg font-bold flex items-center gap-2">
            Model Pipeline Deep-Dive
          </h2>
          <p className="text-sm text-textMuted mt-1">Analysis for {asset.id} - {asset.substation}</p>
        </div>
      </div>

      <div className="flex-1 overflow-auto p-5 space-y-6">
        {/* Model 1: Isolation Forest */}
        <section>
          <h3 className="text-sm font-semibold text-textMuted uppercase tracking-wider mb-4 flex items-center gap-2">
            <ShieldAlert className="w-4 h-4" /> Model 1: Isolation Forest (Anomaly)
          </h3>
          <div className="bg-slate-50 rounded-lg p-4 border border-slate-200 flex items-center gap-6 shadow-sm">
            <div className="relative w-24 h-24 flex-shrink-0">
              <svg className="w-full h-full transform -rotate-90" viewBox="0 0 100 100">
                <circle 
                  cx="50" cy="50" r="40" 
                  stroke="currentColor" 
                  strokeWidth="8" 
                  fill="transparent" 
                  className="text-slate-200"
                />
                <circle 
                  cx="50" cy="50" r="40" 
                  stroke={strokeColor} 
                  strokeWidth="8" 
                  fill="transparent"
                  strokeDasharray={`${anomalyPercentage * 2.51} 251`}
                  className="transition-all duration-1000 ease-out"
                />
              </svg>
              <div className="absolute inset-0 flex flex-col items-center justify-center">
                <span className="text-xl font-bold" style={{ color: strokeColor }}>{asset.model1.anomalyScore.toFixed(2)}</span>
                <span className="text-[10px] text-textMuted font-medium uppercase mt-0.5">Score</span>
              </div>
            </div>
            <div className="flex-1">
              <div className="flex items-center gap-2 mb-2">
                <h4 className="font-medium text-text">Anomaly Flag:</h4>
                <span className={`px-2 py-0.5 rounded text-xs font-bold ${asset.model1.anomalyFlag === -1 ? 'bg-danger/20 text-danger' : 'bg-success/20 text-success'}`}>
                  {asset.model1.anomalyFlag}
                </span>
              </div>
              <div className="mt-3 space-y-1">
                <p className="text-xs font-medium text-textMuted uppercase tracking-wider mb-1">Anomalous Features Triggered:</p>
                <ul className="space-y-1">
                  {asset.model1.anomalousFeatures.map((feature, i) => (
                    <li key={i} className="text-sm text-text flex items-start gap-2">
                      <AlertTriangle className="w-3.5 h-3.5 text-warning flex-shrink-0 mt-0.5" />
                      <span>{feature}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        </section>

        {/* Model 2: XGBoost Regressor */}
        <section>
          <h3 className="text-sm font-semibold text-textMuted uppercase tracking-wider mb-4 flex items-center gap-2">
            <TrendingUp className="w-4 h-4" /> Model 2: XGBoost Regressor (Thermal Delta)
          </h3>
          <div className="bg-slate-50 rounded-lg p-4 border border-slate-200 shadow-sm space-y-4">
            <div className="grid grid-cols-3 gap-3">
              <div className="bg-white p-3 rounded border border-slate-200 flex flex-col items-center justify-center text-center">
                <Thermometer className="w-4 h-4 text-danger mb-1" />
                <span className="text-xs text-textMuted">Ambient</span>
                <span className="font-semibold text-text">{asset.model2.weatherContext.ambientTemp}°C</span>
              </div>
              <div className="bg-white p-3 rounded border border-slate-200 flex flex-col items-center justify-center text-center">
                <Droplets className="w-4 h-4 text-primary mb-1" />
                <span className="text-xs text-textMuted">Humidity</span>
                <span className="font-semibold text-text">{asset.model2.weatherContext.humidity}%</span>
              </div>
              <div className="bg-white p-3 rounded border border-slate-200 flex flex-col items-center justify-center text-center">
                <Wind className="w-4 h-4 text-textMuted mb-1" />
                <span className="text-xs text-textMuted">Wind</span>
                <span className="font-semibold text-text">{asset.model2.weatherContext.windSpeed} m/s</span>
              </div>
            </div>
            
            <div className="h-56 mt-2">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={asset.model2.temperatureData} margin={{ top: 5, right: 20, bottom: 5, left: -20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#E8E5DF" vertical={false} />
                  <XAxis dataKey="time" stroke="#8B899C" fontSize={12} tickLine={false} axisLine={false} />
                  <YAxis stroke="#8B899C" fontSize={12} tickLine={false} axisLine={false} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#FFFFFF', borderColor: '#E8E5DF', color: '#2F2E41', borderRadius: '0.5rem', boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)' }}
                    itemStyle={{ color: '#2F2E41' }}
                  />
                  <Legend iconType="circle" wrapperStyle={{ fontSize: '12px', paddingTop: '10px' }} />
                  <Line type="monotone" name="Predicted Baseline Temp (°C)" dataKey="predicted" stroke="#81B29A" strokeWidth={2} dot={false} />
                  <Line type="monotone" name="Actual Sensor Temp (°C)" dataKey="actual" stroke="#E07A5F" strokeWidth={2} dot={{ r: 4, fill: '#E07A5F', strokeWidth: 0 }} activeDot={{ r: 6 }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </section>

        {/* Model 3: Weighted Scoring Algorithm */}
        <section>
          <h3 className="text-sm font-semibold text-textMuted uppercase tracking-wider mb-4 flex items-center gap-2">
            <Zap className="w-4 h-4" /> Model 3: Risk Scaling Breakdown
          </h3>
          <div className="space-y-3">
            <div className="bg-slate-50 rounded-lg p-4 border border-slate-200 shadow-sm flex items-center justify-between">
              <div className="flex-1 text-center border-r border-slate-200 px-2">
                <div className="text-xs text-textMuted uppercase tracking-wider mb-1">Combined Failure Prob</div>
                <div className="text-xl font-bold text-text">{asset.model3.failureProbabilityScore} <span className="text-sm font-normal text-textMuted">× 0.6</span></div>
              </div>
              <div className="px-4 text-xl font-bold text-textMuted">+</div>
              <div className="flex-1 text-center border-r border-slate-200 px-2">
                <div className="text-xs text-textMuted uppercase tracking-wider mb-1">Severity / Impact</div>
                <div className="text-xl font-bold text-text">{asset.model3.severityScore} <span className="text-sm font-normal text-textMuted">× 0.4</span></div>
              </div>
              <div className="px-4 text-xl font-bold text-textMuted">=</div>
              <div className="flex-1 text-center px-2">
                <div className="text-xs text-danger uppercase font-bold tracking-wider mb-1">Final Risk Index</div>
                <div className="text-2xl font-black text-danger">{asset.riskIndex}</div>
              </div>
            </div>
            
            <div className="bg-slate-50 rounded-lg p-4 border border-slate-200 shadow-sm">
              <h4 className="text-sm font-semibold text-text mb-3">Community Impact Variables (Severity Context)</h4>
              <ul className="space-y-3">
                <li className="flex items-center justify-between text-sm">
                  <span className="text-textMuted">Critical Hospital Connected:</span>
                  <span className={`px-2 py-0.5 rounded font-medium ${asset.model3.impactVariables.hospitalConnected ? 'bg-danger/10 text-danger' : 'bg-slate-100 text-textMuted'}`}>
                    {asset.model3.impactVariables.hospitalConnected ? 'Yes' : 'No'}
                  </span>
                </li>
                <li className="flex items-center justify-between text-sm">
                  <span className="text-textMuted">Critical Water Plant:</span>
                  <span className={`px-2 py-0.5 rounded font-medium ${asset.model3.impactVariables.criticalWaterPlant ? 'bg-danger/10 text-danger' : 'bg-slate-100 text-textMuted'}`}>
                    {asset.model3.impactVariables.criticalWaterPlant ? 'Yes' : 'No'}
                  </span>
                </li>
                <li className="flex items-center justify-between text-sm">
                  <span className="text-textMuted">Homes Powered:</span>
                  <span className="font-semibold text-text">
                    {asset.model3.impactVariables.homesPowered.toLocaleString()}
                  </span>
                </li>
              </ul>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}
