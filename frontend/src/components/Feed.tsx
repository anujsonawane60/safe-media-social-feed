'use client';

import { useState, useEffect } from 'react';
import { Loader2, ImageOff, RefreshCw } from 'lucide-react';
import PostCard from './PostCard';
import { getPosts, Post } from '@/lib/api';

interface FeedProps {
  refreshTrigger: number;
}

export default function Feed({ refreshTrigger }: FeedProps) {
  const [posts, setPosts] = useState<Post[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchPosts = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await getPosts();
      setPosts(data);
    } catch (err: any) {
      console.error('Failed to fetch posts:', err);
      setError(err.message || 'Failed to load posts. Make sure the backend is running.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchPosts();
  }, [refreshTrigger]);

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center py-20">
        <Loader2 className="w-10 h-10 text-primary animate-spin" />
        <p className="mt-4 text-secondary">Loading feed...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center py-20 px-4">
        <div className="bg-red-50 text-red-800 p-6 rounded-xl text-center max-w-md">
          <ImageOff className="w-12 h-12 mx-auto mb-4 opacity-50" />
          <h3 className="font-semibold text-lg mb-2">Unable to load feed</h3>
          <p className="text-sm mb-4">{error}</p>
          <button
            onClick={fetchPosts}
            className="bg-red-100 hover:bg-red-200 text-red-800 px-4 py-2 rounded-lg font-medium transition-colors inline-flex items-center gap-2"
          >
            <RefreshCw className="w-4 h-4" />
            Try again
          </button>
        </div>
      </div>
    );
  }

  if (posts.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-20 px-4">
        <div className="text-center max-w-md">
          <ImageOff className="w-16 h-16 mx-auto mb-4 text-secondary opacity-50" />
          <h3 className="font-semibold text-xl mb-2">No posts yet</h3>
          <p className="text-secondary">
            Be the first to share something! Click the + button above to upload an image or video.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {posts.map((post) => (
        <PostCard key={post.id} post={post} />
      ))}
    </div>
  );
}
