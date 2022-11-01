import React, { useEffect, useState } from "react";
import Button from "./Button";
import axios from "axios";
import { toast } from "react-toastify";

export default function FileUpload() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [file, setFile] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (loading) return;

    setLoading(true);

    let formData = new FormData();
    formData.set("file", file);

    try {
      const res = await axios.post(
        "https://clockomentia.herokuapp.com/predict",
        formData
      );

      if (res && res.data && res.data.status === "success") {
        setResult(res.data);
        toast.success("Predicted successfully");
      } else if (res && res.data && res.data.error) toast.error(res.data.error);
      else toast.error("Error while predicting");

      setFile(null);
      setLoading(false);
    } catch (err) {
      console.log(err);
      toast.error("Some error occured!");
      setLoading(false);
      return;
    }
  };

  return (
    <div className="bg-blue-50">
      <div className="w-full mx-auto px-5 max-w-lg py-12">
        <h2 className="my-10 mx-auto text-center text-3xl font-semibold md:text-4xl">
          Upload File
          <br />
          <span className="text-lg font-medium text-gray-700"></span>
        </h2>

        <form encType="multipart/form-data" onSubmit={handleSubmit}>
          <div className="drag-file-area relative h-72 rounded-lg">
            {loading ? (
              <div className="flex items-center justify-center text-3xl w-full h-full">
                Loading...
              </div>
            ) : file ? (
              <div className="flex items-center justify-center text-blue-500 font-semibold text-lg w-full h-full">
                {file.name.split("\\").slice(-1)}
              </div>
            ) : (
              <>
                <span className="flex justify-center py-4">
                  <svg
                    version="1.1"
                    xmlns="http://www.w3.org/2000/svg"
                    height="40"
                    width="40"
                    viewBox="0 0 512 512"
                  >
                    <g>
                      <g>
                        <path d="m153.7,171.5l81.9-88.1v265.3c0,11.3 9.1,20.4 20.4,20.4 11.3,0 20.4-9.1 20.4-20.4v-265.3l81.9,88.1c7.7,8.3 20.6,8.7 28.9,1.1 8.3-7.7 8.7-20.6 1.1-28.9l-117.3-126.2c-11.5-11.6-25.6-5.2-29.9,0l-117.3,126.2c-7.7,8.3-7.2,21.2 1.1,28.9 8.2,7.6 21.1,7.2 28.8-1.1z" />
                        <path d="M480.6,341.2c-11.3,0-20.4,9.1-20.4,20.4V460H51.8v-98.4c0-11.3-9.1-20.4-20.4-20.4S11,350.4,11,361.6v118.8    c0,11.3,9.1,20.4,20.4,20.4h449.2c11.3,0,20.4-9.1,20.4-20.4V361.6C501,350.4,491.9,341.2,480.6,341.2z" />
                      </g>
                    </g>
                  </svg>
                </span>
                <h3 className="dynamic-message">Drag & drop any file here </h3>
                <label className="label">
                  or
                  <input
                    type="file"
                    id="fileInput"
                    onChange={(e) => {
                      setFile(e.target.files[0]);
                    }}
                    accept={["jpeg", "png", "jpg"]}
                    className="default-file-input h-20 opacity-0 bg-red-500"
                  />
                  <div className="relative top-[-50px] ">
                    <span className="text-blue-600 hover:cursor-pointer font-semibold">
                      Browse file
                    </span>{" "}
                    <span>from device</span>
                  </div>
                </label>
              </>
            )}
          </div>
          <Button
            type="submit"
            disabled={loading || !file}
            className="mx-auto my-8 w-full py-3 px-6 disabled:cursor-not-allowed bg-blue-500 flex items-center justify-center hover:bg-blue-600 text-white
              rounded-xl"
          >
            Upload
          </Button>
        </form>
        {result && (
          <div className="pt-6">
            <h2 className="mx-auto mb-4 text-center text-xl font-semibold md:text-2xl">
              Result
            </h2>
            <div className="flex">
              <div className="border rounded-tl-md font-semibold bg-blue-500 text-white border-black flex-1 text-center py-1">
                S. No.
              </div>
              <div className="border rounded-tr-md font-semibold bg-blue-500 text-white border-black flex-1 text-center py-1">
                probability
              </div>
            </div>
            {result.probabilities.split(" ").map((value, i) => (
              <div key={i} className="flex bg-white">
                <div className="border border-black flex-1 text-center py-1">
                  {i + 1}
                </div>
                <div className="border border-black flex-1 text-center py-1">
                  {value}
                </div>
              </div>
            ))}
            <div className="flex">
              <div className="border rounded-bl-md font-semibold bg-blue-500 text-white border-black flex-1 text-center py-1">
                Prediction value
              </div>
              <div className="border rounded-br-md font-semibold bg-blue-500 text-white border-black flex-1 text-center py-1">
                {result.prediction}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
