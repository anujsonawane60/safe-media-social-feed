'use client';

import { Home, PlusSquare, Shield } from 'lucide-react';

interface HeaderProps {
  onUploadClick: () => void;
}

export default function Header({ onUploadClick }: HeaderProps) {
  return (
    <header className="fixed top-0 left-0 right-0 bg-white border-b border-border z-50">
      <div className="max-w-5xl mx-auto px-4 h-16 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Shield className="w-8 h-8 text-primary" />
          <h1 className="text-xl font-semibold hidden sm:block">SafeFeed</h1>
        </div>

        <nav className="flex items-center gap-6">
          <button className="p-2 hover:opacity-60 transition-opacity">
            <Home className="w-6 h-6" />
          </button>
          <button
            onClick={onUploadClick}
            className="p-2 hover:opacity-60 transition-opacity"
            aria-label="Create new post"
          >
            <PlusSquare className="w-6 h-6" />
          </button>
        </nav>
      </div>
    </header>
  );
}
