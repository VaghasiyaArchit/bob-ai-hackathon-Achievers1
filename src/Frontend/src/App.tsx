import { useState } from 'react';
import { TopNav } from './components/TopNav';
import { StatsBar } from './components/StatsBar';
import { AssetTable } from './components/AssetTable';
import { AssetInsights } from './components/AssetInsights';
import { type Asset, mockAssets } from './data/mockData';
import { CheckCircle2, X } from 'lucide-react';
import { LoadingScreen } from './components/LoadingScreen';

function App() {
  const [isLoading, setIsLoading] = useState(true);
  const [selectedAsset, setSelectedAsset] = useState<Asset | null>(mockAssets[0]);
  const [alertMessage, setAlertMessage] = useState<string | null>(null);

  const handleDispatchSuccess = (message: string) => {
    setAlertMessage(message);
    setTimeout(() => {
      setAlertMessage(null);
    }, 4000);
  };

  return (
    <>
      {isLoading && <LoadingScreen onLoadingComplete={() => setIsLoading(false)} />}
      
      <div className={`min-h-screen bg-background flex flex-col transition-opacity duration-1000 ${isLoading ? 'opacity-0 h-0 overflow-hidden' : 'opacity-100'}`}>
        <TopNav />
      
      <main className="flex-1 p-6 overflow-hidden flex flex-col">
        <StatsBar />
        
        <div className="flex-1 flex gap-6 min-h-0">
          <div className="w-[60%] flex flex-col min-h-0">
            <AssetTable 
              assets={mockAssets}
              onSelectAsset={setSelectedAsset} 
              selectedAssetId={selectedAsset?.id}
              onDispatchSuccess={handleDispatchSuccess}
            />
          </div>
          <div className="w-[40%] flex flex-col min-h-0">
            <AssetInsights asset={selectedAsset} />
          </div>
        </div>
      </main>

      {/* Success Alert Toast */}
      {alertMessage && (
        <div className="fixed bottom-6 right-6 bg-emerald-50 border border-emerald-200 text-emerald-800 px-4 py-3 rounded-lg shadow-lg flex items-center gap-3 animate-in slide-in-from-bottom-5">
          <CheckCircle2 className="w-5 h-5 text-emerald-600" />
          <span className="font-medium text-sm">{alertMessage}</span>
          <button 
            onClick={() => setAlertMessage(null)}
            className="ml-4 p-1 hover:bg-emerald-100 rounded text-emerald-600 transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      )}
      </div>
    </>
  );
}

export default App;
