'use client';

import { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { X, Upload, Image, Film, AlertTriangle, CheckCircle, Loader2 } from 'lucide-react';
import { uploadMedia } from '@/lib/api';

interface UploadModalProps {
  isOpen: boolean;
  onClose: () => void;
  onUploadSuccess: () => void;
}

type UploadState = 'idle' | 'uploading' | 'success' | 'blocked' | 'error';

export default function UploadModal({ isOpen, onClose, onUploadSuccess }: UploadModalProps) {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [caption, setCaption] = useState('');
  const [uploadState, setUploadState] = useState<UploadState>('idle');
  const [message, setMessage] = useState('');
  const [nsfwScore, setNsfwScore] = useState<number | null>(null);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const selectedFile = acceptedFiles[0];
    if (selectedFile) {
      setFile(selectedFile);
      setUploadState('idle');
      setMessage('');
      setNsfwScore(null);

      // Create preview
      const reader = new FileReader();
      reader.onload = () => {
        setPreview(reader.result as string);
      };
      reader.readAsDataURL(selectedFile);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'],
      'video/*': ['.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v'],
    },
    maxFiles: 1,
    multiple: false,
  });

  const handleUpload = async () => {
    if (!file) return;

    setUploadState('uploading');
    setMessage('Analyzing content for safety...');

    try {
      const response = await uploadMedia(file, caption || undefined);

      if (response.success) {
        setUploadState('success');
        setMessage(response.message);
        setNsfwScore(response.nsfw_score ?? null);

        // Reset and close after delay
        setTimeout(() => {
          resetForm();
          onClose();
          onUploadSuccess();
        }, 1500);
      } else {
        setUploadState('blocked');
        setMessage(response.message);
        setNsfwScore(response.nsfw_score ?? null);
      }
    } catch (error: any) {
      setUploadState('error');
      setMessage(error.response?.data?.detail || 'An error occurred during upload');
    }
  };

  const resetForm = () => {
    setFile(null);
    setPreview(null);
    setCaption('');
    setUploadState('idle');
    setMessage('');
    setNsfwScore(null);
  };

  const handleClose = () => {
    resetForm();
    onClose();
  };

  const isVideo = file?.type.startsWith('video/');

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-xl w-full max-w-lg max-h-[90vh] overflow-hidden animate-fade-in">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-border">
          <h2 className="text-lg font-semibold">Create new post</h2>
          <button
            onClick={handleClose}
            className="p-1 hover:bg-gray-100 rounded-full transition-colors"
          >
            <X className="w-6 h-6" />
          </button>
        </div>

        {/* Content */}
        <div className="p-4 overflow-y-auto max-h-[calc(90vh-140px)]">
          {!file ? (
            /* Dropzone */
            <div
              {...getRootProps()}
              className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-colors ${
                isDragActive
                  ? 'border-primary bg-primary/5'
                  : 'border-border hover:border-primary/50'
              }`}
            >
              <input {...getInputProps()} />
              <div className="flex flex-col items-center gap-4">
                <div className="flex gap-2">
                  <Image className="w-12 h-12 text-secondary" />
                  <Film className="w-12 h-12 text-secondary" />
                </div>
                <div>
                  <p className="text-lg font-medium">Drag photos and videos here</p>
                  <p className="text-secondary text-sm mt-1">or click to select files</p>
                </div>
                <button className="bg-primary text-white px-4 py-2 rounded-lg font-medium hover:bg-primary/90 transition-colors">
                  Select from computer
                </button>
              </div>
            </div>
          ) : (
            /* Preview and Caption */
            <div className="space-y-4">
              {/* Preview */}
              <div className="relative aspect-square bg-gray-100 rounded-xl overflow-hidden">
                {isVideo ? (
                  <video
                    src={preview || undefined}
                    className="w-full h-full object-contain"
                    controls
                  />
                ) : (
                  <img
                    src={preview || undefined}
                    alt="Preview"
                    className="w-full h-full object-contain"
                  />
                )}

                {uploadState === 'idle' && (
                  <button
                    onClick={resetForm}
                    className="absolute top-2 right-2 p-1.5 bg-black/50 rounded-full text-white hover:bg-black/70 transition-colors"
                  >
                    <X className="w-4 h-4" />
                  </button>
                )}
              </div>

              {/* Caption */}
              {uploadState === 'idle' && (
                <textarea
                  value={caption}
                  onChange={(e) => setCaption(e.target.value)}
                  placeholder="Write a caption..."
                  className="w-full p-3 border border-border rounded-xl resize-none focus:outline-none focus:border-primary transition-colors"
                  rows={3}
                  maxLength={2200}
                />
              )}

              {/* Status Messages */}
              {message && (
                <div
                  className={`p-4 rounded-xl flex items-start gap-3 ${
                    uploadState === 'success'
                      ? 'bg-green-50 text-green-800'
                      : uploadState === 'blocked'
                      ? 'bg-red-50 text-red-800'
                      : uploadState === 'uploading'
                      ? 'bg-blue-50 text-blue-800'
                      : 'bg-red-50 text-red-800'
                  }`}
                >
                  {uploadState === 'success' && <CheckCircle className="w-5 h-5 flex-shrink-0 mt-0.5" />}
                  {uploadState === 'blocked' && <AlertTriangle className="w-5 h-5 flex-shrink-0 mt-0.5" />}
                  {uploadState === 'uploading' && <Loader2 className="w-5 h-5 flex-shrink-0 mt-0.5 animate-spin" />}
                  {uploadState === 'error' && <AlertTriangle className="w-5 h-5 flex-shrink-0 mt-0.5" />}
                  <div>
                    <p className="font-medium">{message}</p>
                    {nsfwScore !== null && uploadState === 'success' && (
                      <p className="text-sm mt-1 opacity-75">
                        Safety score: {((1 - nsfwScore) * 100).toFixed(1)}%
                      </p>
                    )}
                  </div>
                </div>
              )}

              {/* Upload Button */}
              {uploadState === 'idle' && (
                <button
                  onClick={handleUpload}
                  className="w-full bg-primary text-white py-3 rounded-xl font-semibold hover:bg-primary/90 transition-colors flex items-center justify-center gap-2"
                >
                  <Upload className="w-5 h-5" />
                  Share
                </button>
              )}

              {/* Try Again Button for blocked/error */}
              {(uploadState === 'blocked' || uploadState === 'error') && (
                <button
                  onClick={resetForm}
                  className="w-full bg-gray-200 text-gray-800 py-3 rounded-xl font-semibold hover:bg-gray-300 transition-colors"
                >
                  Try with different content
                </button>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
