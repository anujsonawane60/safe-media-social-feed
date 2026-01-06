'use client';

import { useState } from 'react';
import { Heart, MessageCircle, Share2, Bookmark, MoreHorizontal, Shield } from 'lucide-react';
import { formatDistanceToNow } from 'date-fns';
import { Post, getMediaUrl } from '@/lib/api';

interface PostCardProps {
  post: Post;
}

export default function PostCard({ post }: PostCardProps) {
  const [liked, setLiked] = useState(false);
  const [saved, setSaved] = useState(false);
  const [likesCount, setLikesCount] = useState(Math.floor(Math.random() * 1000));

  const mediaUrl = getMediaUrl(post.file_url);
  const timeAgo = formatDistanceToNow(new Date(post.created_at), { addSuffix: true });
  const safetyPercentage = ((1 - post.nsfw_score) * 100).toFixed(0);

  const handleLike = () => {
    setLiked(!liked);
    setLikesCount(prev => liked ? prev - 1 : prev + 1);
  };

  return (
    <article className="bg-white border border-border rounded-lg overflow-hidden animate-fade-in">
      {/* Header */}
      <div className="flex items-center justify-between p-3">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-full bg-gradient-to-br from-primary to-purple-500 flex items-center justify-center">
            <span className="text-white text-sm font-medium">U</span>
          </div>
          <div>
            <p className="text-sm font-semibold">user_{post.id.slice(0, 6)}</p>
            <p className="text-xs text-secondary">{timeAgo}</p>
          </div>
        </div>
        <button className="p-2 hover:bg-gray-100 rounded-full transition-colors">
          <MoreHorizontal className="w-5 h-5" />
        </button>
      </div>

      {/* Media */}
      <div className="relative bg-black aspect-square flex items-center justify-center">
        {post.media_type === 'video' ? (
          <video
            src={mediaUrl}
            className="w-full h-full object-contain"
            controls
            playsInline
          />
        ) : (
          <img
            src={mediaUrl}
            alt={post.caption || 'Post image'}
            className="w-full h-full object-contain"
            loading="lazy"
          />
        )}

        {/* Safety Badge */}
        <div className="absolute top-3 right-3 bg-green-500/90 text-white px-2 py-1 rounded-full text-xs font-medium flex items-center gap-1">
          <Shield className="w-3 h-3" />
          {safetyPercentage}% Safe
        </div>
      </div>

      {/* Actions */}
      <div className="p-3">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-4">
            <button
              onClick={handleLike}
              className="hover:opacity-60 transition-opacity"
              aria-label={liked ? 'Unlike' : 'Like'}
            >
              <Heart
                className={`w-6 h-6 ${liked ? 'fill-red-500 text-red-500' : ''}`}
              />
            </button>
            <button className="hover:opacity-60 transition-opacity" aria-label="Comment">
              <MessageCircle className="w-6 h-6" />
            </button>
            <button className="hover:opacity-60 transition-opacity" aria-label="Share">
              <Share2 className="w-6 h-6" />
            </button>
          </div>
          <button
            onClick={() => setSaved(!saved)}
            className="hover:opacity-60 transition-opacity"
            aria-label={saved ? 'Unsave' : 'Save'}
          >
            <Bookmark className={`w-6 h-6 ${saved ? 'fill-black' : ''}`} />
          </button>
        </div>

        {/* Likes count */}
        <p className="text-sm font-semibold mb-1">{likesCount.toLocaleString()} likes</p>

        {/* Caption */}
        {post.caption && (
          <p className="text-sm">
            <span className="font-semibold mr-2">user_{post.id.slice(0, 6)}</span>
            {post.caption}
          </p>
        )}

        {/* Comments placeholder */}
        <button className="text-sm text-secondary mt-2 hover:text-gray-600 transition-colors">
          View all comments
        </button>
      </div>
    </article>
  );
}
