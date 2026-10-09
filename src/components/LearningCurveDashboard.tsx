"use client";

import React, { useState, useEffect } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  AreaChart, Area
} from 'recharts';

interface LearningCurveProps {
  title: string;
  description: string;
  epochs?: number;
  initialLoss?: number;
}

export default function LearningCurveDashboard({ 
  title, 
  description, 
  epochs = 50,
  initialLoss = 2.5
}: LearningCurveProps) {
  const [data, setData] = useState<any[]>([]);
  const [isAnimating, setIsAnimating] = useState(false);

  // Generate realistic learning curves
  useEffect(() => {
    const generateLearningCurve = () => {
      const newData = [];
      for (let i = 0; i <= epochs; i++) {
        // Exponential decay + noise
        const trainLoss = initialLoss * Math.exp(-i / 15) + (Math.random() * 0.1 - 0.05);
        const valLoss = initialLoss * Math.exp(-i / 12) + (Math.random() * 0.15 - 0.075);
        const accuracy = (1 - Math.exp(-i / 8)) * 100 + (Math.random() * 2 - 1);
        
        newData.push({
          epoch: i,
          trainLoss: Math.max(0.01, trainLoss),
          valLoss: Math.max(0.01, valLoss),
          accuracy: Math.min(100, Math.max(0, accuracy))
        });
      }
      setData(newData);
    };

    generateLearningCurve();
  }, [epochs, initialLoss]);

  return (
    <div className="bg-gradient-to-br from-slate-50 to-slate-100 rounded-2xl p-8 my-8 not-prose border border-slate-200">
      <div className="mb-8">
        <h3 className="text-2xl font-bold text-slate-900 mb-2">{title}</h3>
        <p className="text-slate-600">{description}</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Loss Curve */}
        <div className="bg-white rounded-xl p-6 shadow-sm border border-slate-100">
          <h4 className="font-bold text-slate-800 mb-4 flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-red-500"></span>
            Training Loss Over Time
          </h4>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={data}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="epoch" label={{ value: 'Epoch', position: 'insideBottomRight', offset: -5 }} />
              <YAxis label={{ value: 'Loss', angle: -90, position: 'insideLeft' }} />
              <Tooltip 
                contentStyle={{ backgroundColor: '#1e293b', border: 'none', borderRadius: '8px', color: '#fff' }}
              />
              <Legend />
              <Line 
                type="monotone" 
                dataKey="trainLoss" 
                stroke="#ef4444" 
                name="Train Loss"
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
              />
              <Line 
                type="monotone" 
                dataKey="valLoss" 
                stroke="#f97316" 
                name="Validation Loss"
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Accuracy Curve */}
        <div className="bg-white rounded-xl p-6 shadow-sm border border-slate-100">
          <h4 className="font-bold text-slate-800 mb-4 flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-emerald-500"></span>
            Model Accuracy Growth
          </h4>
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={data}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="epoch" label={{ value: 'Epoch', position: 'insideBottomRight', offset: -5 }} />
              <YAxis domain={[0, 100]} label={{ value: 'Accuracy (%)', angle: -90, position: 'insideLeft' }} />
              <Tooltip 
                contentStyle={{ backgroundColor: '#1e293b', border: 'none', borderRadius: '8px', color: '#fff' }}
                formatter={(value) => `${(value as number).toFixed(2)}%`}
              />
              <Area 
                type="monotone" 
                dataKey="accuracy" 
                fill="#10b981" 
                stroke="#059669"
                strokeWidth={2}
                isAnimationActive={false}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Key Insights */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-8">
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
          <div className="text-sm text-blue-600 font-semibold mb-1">Convergence Point</div>
          <div className="text-2xl font-bold text-blue-900">{(epochs * 0.7).toFixed(0)} epochs</div>
          <div className="text-xs text-blue-600 mt-2">Typical convergence threshold</div>
        </div>
        
        <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
          <div className="text-sm text-purple-600 font-semibold mb-1">Final Accuracy</div>
          <div className="text-2xl font-bold text-purple-900">
            {data[data.length - 1]?.accuracy.toFixed(1)}%
          </div>
          <div className="text-xs text-purple-600 mt-2">Achieved on last epoch</div>
        </div>

        <div className="bg-amber-50 border border-amber-200 rounded-lg p-4">
          <div className="text-sm text-amber-600 font-semibold mb-1">Overfitting Risk</div>
          <div className="text-2xl font-bold text-amber-900">
            {data[data.length - 1] ? 
              ((data[data.length - 1].valLoss - data[data.length - 1].trainLoss) * 100).toFixed(1) + '%'
              : '0%'
            }
          </div>
          <div className="text-xs text-amber-600 mt-2">Val loss - Train loss gap</div>
        </div>
      </div>
    </div>
  );
}
