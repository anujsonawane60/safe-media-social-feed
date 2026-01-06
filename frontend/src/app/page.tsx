'use client';

import { useState } from 'react';
import Header from '@/components/Header';
import Feed from '@/components/Feed';
import UploadModal from '@/components/UploadModal';
import { Shield } from 'lucide-react';

export default function Home() {
  const [isUploadModalOpen, setIsUploadModalOpen] = useState(false);
  const [refreshTrigger, setRefreshTrigger] = useState(0);

  const handleUploadSuccess = () => {
    setRefreshTrigger(prev => prev + 1);
  };

  return (
    <div className="min-h-screen bg-background">
      <Header onUploadClick={() => setIsUploadModalOpen(true)} />

      <main className="pt-20 pb-8">
        <div className="max-w-lg mx-auto px-4">
          {/* Welcome Banner */}
          <div className="bg-gradient-to-r from-primary to-purple-500 text-white p-4 rounded-xl mb-6 shadow-lg">
            <div className="flex items-center gap-3">
              <Shield className="w-10 h-10" />
              <div>
                <h2 className="font-bold text-lg">SafeFeed</h2>
                <p className="text-sm opacity-90">
                  AI-powered content moderation keeps this feed safe for everyone
                </p>
              </div>
            </div>
          </div>

          {/* Feed */}
          <Feed refreshTrigger={refreshTrigger} />
        </div>
      </main>

      {/* Upload Modal */}
      <UploadModal
        isOpen={isUploadModalOpen}
        onClose={() => setIsUploadModalOpen(false)}
        onUploadSuccess={handleUploadSuccess}
      />
    </div>
  );
}
