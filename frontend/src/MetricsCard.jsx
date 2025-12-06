import React, { useEffect, useState } from "react";

const MetricsCard = () => {
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("http://127.0.0.1:5000/metrics")
      .then((res) => res.json())
      .then((data) => {
        setMetrics(data);
        setLoading(false);
      })
      .catch((err) => {
        console.error("Error fetching metrics:", err);
        setLoading(false);
      });
  }, []);

  if (loading) {
    return (
      <div className="p-6 bg-white shadow-md rounded-xl w-96 text-center">
        <p className="text-gray-500">Loading metrics...</p>
      </div>
    );
  }

  if (!metrics || metrics.error) {
    return (
      <div className="p-6 bg-red-100 shadow-md rounded-xl w-96 text-center">
        <p className="text-red-600">No metrics available</p>
      </div>
    );
  }

  return (
    <div className="p-6 bg-white shadow-lg rounded-2xl w-96 border border-gray-200">
      <h2 className="text-xl font-bold mb-4 text-gray-800">📊 Model Metrics</h2>
      <ul className="space-y-2">
        {Object.entries(metrics).map(([key, value]) => (
          <li
            key={key}
            className="flex justify-between text-gray-700 border-b pb-1"
          >
            <span className="capitalize">{key}</span>
            <span className="font-semibold">{value}</span>
          </li>
        ))}
      </ul>
    </div>
  );
};

export default MetricsCard;
