import { useEffect, useState } from "react";
import axiosInstance from "@/lib/axios";

type ImageEntry = {
  url: string;
  filename: string;
  key: string;
};

type PersonImages = {
  personId: number;
  images: ImageEntry[];
};

const GroupedImages = () => {
  const [personImageGroups, setPersonImageGroups] = useState<PersonImages[]>([]);
  const [openFolders, setOpenFolders] = useState<number[]>([]);

  useEffect(() => {
    const fetchImages = async () => {
      try {
        const res = await axiosInstance.get("/api/images/grouped");
        const data = res.data;
        const groups: PersonImages[] = Object.keys(data).map((id) => ({
          personId: parseInt(id),
          images: data[id],
        }));
        setPersonImageGroups(groups);
      } catch (err) {
        console.error("Failed to load grouped images", err);
      }
    };

    fetchImages();
  }, []);

  const toggleFolder = (personId: number) => {
    setOpenFolders((prev) =>
      prev.includes(personId)
        ? prev.filter((id) => id !== personId)
        : [...prev, personId]
    );
  };

  const downloadZip = (personId: number) => {
    const link = document.createElement("a");
    link.href = `/api/images/download_group/${personId}`;
    link.download = `person_${personId}_images.zip`;
    link.click();
  };

  return (
    <div className="p-8">
      {personImageGroups.map(({ personId, images }) => {
        const isOpen = openFolders.includes(personId);

        return (
          <div key={personId} className="border border-gray-300 rounded-lg mb-4">
            <div
              onClick={() => toggleFolder(personId)}
              className="cursor-pointer bg-gray-100 p-4 flex justify-between items-center"
            >
              <h3 className="m-0">Person {personId}</h3>
              <span>{isOpen ? "▼" : "▶"}</span>
            </div>

            {isOpen && (
              <div className="p-4">
                <button
                  onClick={() => downloadZip(personId)}
                  className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 transition"
                >
                  Download All
                </button>
                <div className="flex flex-wrap gap-4 mt-4">
                  {images.map((img) => (
                    <div key={img.key} className="flex flex-col">
                      <img
                        src={img.url}
                        alt={img.filename}
                        className="w-[150px] rounded"
                      />
                      <p className="text-sm text-gray-600 mt-1">{img.filename}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
};

export default GroupedImages;