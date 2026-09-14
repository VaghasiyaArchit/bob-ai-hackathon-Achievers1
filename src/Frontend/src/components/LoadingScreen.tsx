import { Activity } from 'lucide-react';
import BlurText from './BlurText';

interface LoadingScreenProps {
  onLoadingComplete: () => void;
}

export function LoadingScreen({ onLoadingComplete }: LoadingScreenProps) {
  return (
    <div className="fixed inset-0 bg-background flex flex-col items-center justify-center z-50">
      <div className="flex flex-col items-center">
        {/* Logo Section */}
        <div className="p-4 bg-primary/10 rounded-2xl mb-6 shadow-[0_0_30px_rgba(59,130,246,0.3)] animate-pulse">
          <Activity className="w-16 h-16 text-primary" />
        </div>
        
        {/* Animated Text */}
        <BlurText
          text="GridPulse AI"
          delay={400}
          animateBy="words"
          direction="top"
          onAnimationComplete={() => {
            // Add a longer delay after animation finishes before transitioning
            setTimeout(() => {
              onLoadingComplete();
            }, 2000);
          }}
          className="text-4xl font-bold tracking-tight text-text mb-4"
        />
        
        <p className="text-textMuted text-sm tracking-widest uppercase mt-4">
          Initializing Predictive Models...
        </p>
      </div>
    </div>
  );
}
